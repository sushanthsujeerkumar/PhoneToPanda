"""Calibrate two orientation sensors and draw a shoulder-relative arm.

Quaternions are Hamilton wxyz active rotations. +X is forward, +Y is left,
and +Z is up; a neutral arm points down along -Z. Table capture aligns each
sensor's world yaw on the LEFT of its quaternion. Mounted-neutral capture
puts the mounting offset on the RIGHT: segment = alignment * raw * offset.
Keeping these corrections separate avoids rotating subsequent motion axes.

The shoulder stays at the origin. Elbow and wrist positions follow configured
segment lengths; acceleration is never integrated into position. Torso motion,
mount slip and heading drift are not independently observed by these two IMUs.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np

from .protocol import normalize_sensor, sensor_id

IDENTITY = np.array([1., 0., 0., 0.])
TABLE_SECONDS = 2.0
NEUTRAL_SECONDS = 3.0
MIN_CAPTURE_SAMPLES = 20
MAX_MOVEMENT_DEG = 3.0
MAX_GYRO_RAD_S = .10
TABLE_MAX_TILT_DEG = 15.0


def normalize(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Quaternion must contain four finite values, wxyz")
    norm = float(np.linalg.norm(q))
    if norm < 1e-10:
        raise ValueError("Quaternion has zero length")
    return q / norm


def multiply(a, b):
    aw, ax, ay, az = normalize(a)
    bw, bx, by, bz = normalize(b)
    return normalize([aw*bw - ax*bx - ay*by - az*bz,
                      aw*bx + ax*bw + ay*bz - az*by,
                      aw*by - ax*bz + ay*bw + az*bx,
                      aw*bz + ax*by - ay*bx + az*bw])


def inverse(q):
    w, x, y, z = normalize(q)
    return np.array([w, -x, -y, -z])


def rotate(q, vector):
    q = normalize(q)
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Vector must contain three finite values")
    return vector + 2 * np.cross(q[1:], np.cross(q[1:], vector) + q[0] * vector)


def axis_angle(axis, radians):
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isfinite(radians) or np.linalg.norm(axis) < 1e-10:
        raise ValueError("Rotation requires a finite nonzero axis and angle")
    return np.r_[math.cos(radians / 2), axis / np.linalg.norm(axis) * math.sin(radians / 2)]


def angle_between(a, b):
    return 2 * math.acos(float(np.clip(abs(np.dot(normalize(a), normalize(b))), 0, 1)))


def average(quaternions):
    """Average orientations without treating q and -q as different rotations."""
    qs = np.array([normalize(q) for q in quaternions])
    if not len(qs):
        raise ValueError("Cannot average an empty capture")
    _, eigenvectors = np.linalg.eigh(qs.T @ qs)
    result = eigenvectors[:, -1]
    return normalize(result if np.dot(result, qs[0]) >= 0 else -result)


def _clock(now):
    result = time.monotonic() if now is None else float(now)
    if not math.isfinite(result):
        raise ValueError("Host time must be finite monotonic seconds")
    return result


def _length(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .05 <= value <= 1.:
        raise ValueError("Each segment length must be between 0.05 and 1.0 metres")
    return float(value)


class ArmMotion:
    """Small state machine; call update for frames and pose/status for UI refreshes.

    Call invalidate whenever the connection, device orientation generation or
    physical sensor mounting changes. Mapping changes invalidate automatically.
    Repeated timestamps cannot complete a capture or keep a frozen stream fresh.
    """
    def __init__(self, upper_id="3", forearm_id="4", upper_length_m=.30,
                 forearm_length_m=.26, stale_after_s=.25):
        self.mapping = {"upper": sensor_id(upper_id), "forearm": sensor_id(forearm_id)}
        if len(set(self.mapping.values())) != 2:
            raise ValueError("Upper arm and forearm must use different sensor IDs")
        self.lengths = {"upper": _length(upper_length_m), "forearm": _length(forearm_length_m)}
        if not isinstance(stale_after_s, (int, float)) or not math.isfinite(stale_after_s) or stale_after_s <= 0:
            raise ValueError("Freshness timeout must be positive seconds")
        self.stale_after_s = float(stale_after_s)
        self.generation = 0
        self.calibration_version = 0
        self._source_generation = None
        self._last_frame_t = None
        self._latest = {}
        self._history = {}
        self._last_clock = None
        self._clear_calibration("Connect the sensors, then align them flat on a table.")

    def _clear_calibration(self, message):
        self.alignment = {role: IDENTITY.copy() for role in self.mapping}
        self.offset = {role: IDENTITY.copy() for role in self.mapping}
        self.table_ready = self.neutral_ready = False
        self.stage, self.message = "uncalibrated", str(message)
        self._clear_capture()

    def _clear_capture(self):
        self._samples = []
        self._capture_start = None
        self._capture_pair = None
        self.progress = 0.0

    def invalidate(self, reason="Connection or sensor orientation changed; repeat calibration."):
        self.generation += 1
        self.calibration_version += 1
        self._latest = {}
        self._history = {}
        self._last_frame_t = None
        self._source_generation = None
        self._last_clock = None
        self._clear_calibration(reason)
        return self.status()

    def configure(self, *, upper_id=None, forearm_id=None, upper_length_m=None, forearm_length_m=None):
        mapping = {"upper": self.mapping["upper"] if upper_id is None else sensor_id(upper_id),
                   "forearm": self.mapping["forearm"] if forearm_id is None else sensor_id(forearm_id)}
        if len(set(mapping.values())) != 2:
            raise ValueError("Upper arm and forearm must use different sensor IDs")
        lengths = {"upper": self.lengths["upper"] if upper_length_m is None else _length(upper_length_m),
                   "forearm": self.lengths["forearm"] if forearm_length_m is None else _length(forearm_length_m)}
        changed = mapping != self.mapping
        self.mapping, self.lengths = mapping, lengths
        if changed:
            return self.invalidate("Sensor mapping changed; repeat table and mounted-neutral calibration.")
        return self.status()

    def _health(self, now):
        health = {}
        for identity, sample in self._latest.items():
            elapsed = max(0., now - sample["_received_at"])
            unchanged = max(0., now - sample["_advanced_at"])
            age = sample["age_ms"] / 1000 + elapsed
            reason = sample["reason"]
            if age >= self.stale_after_s or unchanged >= self.stale_after_s:
                reason = "No fresh sensor sample; pose is paused"
            health[identity] = {"id": identity, "name": sample["name"],
                                "ok": bool(sample["ok"] and not reason), "reason": reason,
                                "age_ms": float(max(age, unchanged) * 1000), "flags": sample["flags"],
                                "t_us": sample["t_us"], "errors": sample.get("errors", 0),
                                "has_gyro": sample["g"] is not None, "has_accel": sample["a"] is not None}
        for identity in self.mapping.values():
            health.setdefault(identity, {"id": identity, "name": identity, "ok": False,
                                        "reason": "Selected sensor is missing", "age_ms": None,
                                        "t_us": None, "flags": 0, "errors": 0,
                                        "has_gyro": False, "has_accel": False})
        return health

    def _selected(self, now):
        health = self._health(now)
        for role, identity in self.mapping.items():
            if not health[identity]["ok"]:
                raise ValueError(f"{role.replace('_', ' ')}: {health[identity]['reason']}")
        return {role: self._latest[identity] for role, identity in self.mapping.items()}

    def _stationarity(self, selected):
        for role, sample in selected.items():
            if sample["g"] is not None and np.linalg.norm(sample["g"]) > MAX_GYRO_RAD_S:
                raise ValueError(f"Keep {role} still: gyro exceeds {MAX_GYRO_RAD_S:.2f} rad/s")
            if sample["a"] is not None and not .85 <= np.linalg.norm(sample["a"]) <= 1.15:
                raise ValueError(f"Keep {role} still: acceleration should be near 1 g")
            if self._samples and angle_between(sample["q"], self._samples[0][role]) > math.radians(MAX_MOVEMENT_DEG):
                raise ValueError("Orientation moved more than 3 degrees; hold still to restart capture")
        return True

    def _begin(self, stage, now):
        now = _clock(now)
        if self.stage in ("table", "neutral"):
            raise ValueError("A calibration capture is already running")
        self._selected(now)
        if stage == "neutral" and not self.table_ready:
            raise ValueError("Complete table alignment before mounted-neutral calibration")
        if stage == "table":
            self._clear_calibration("Keep both sensors flat, chip side up, pointing the same way.")
        else:
            self.neutral_ready = False
            self._clear_capture()
            self.message = "Keep the mounted right arm straight down and still."
        self.stage = stage
        self.calibration_version += 1
        return self.status(now)

    def start_table(self, now=None):
        return self._begin("table", now)

    def start_neutral(self, now=None):
        return self._begin("neutral", now)

    def _capture(self, selected, now):
        self._stationarity(selected)
        if self.stage == "table":
            for role, sample in selected.items():
                if rotate(sample["q"], (0, 0, 1))[2] < math.cos(math.radians(TABLE_MAX_TILT_DEG)):
                    raise ValueError(f"Place {role} flat, chip side up (within 15 degrees)")
        pair = tuple(selected[role]["t_us"] for role in ("upper", "forearm"))
        if self._capture_pair is not None and any(new <= old for new, old in zip(pair, self._capture_pair)):
            return
        if self._capture_start is None:
            self._capture_start = now
        self._capture_pair = pair
        self._samples.append({role: normalize(sample["q"]) for role, sample in selected.items()})
        duration = TABLE_SECONDS if self.stage == "table" else NEUTRAL_SECONDS
        self.progress = min(1., max(0., (now - self._capture_start) / duration))
        self.message = (f"Hold still: waiting for {MIN_CAPTURE_SAMPLES} distinct paired samples"
                        if self.progress >= 1. and len(self._samples) < MIN_CAPTURE_SAMPLES else
                        f"Hold still: {self.stage} capture {round(self.progress * 100)}%")
        if self.progress < 1. or len(self._samples) < MIN_CAPTURE_SAMPLES:
            return
        means = {role: average([sample[role] for sample in self._samples]) for role in self.mapping}
        if self.stage == "table":
            for role, q in means.items():
                forward = rotate(q, (1, 0, 0))
                self.alignment[role] = axis_angle((0, 0, 1), -math.atan2(forward[1], forward[0]))
            self.table_ready, self.stage = True, "table_ready"
            self.message = "Table aligned. Mount both sensors; then capture the arm straight down."
        else:
            self.offset = {role: inverse(multiply(self.alignment[role], q)) for role, q in means.items()}
            self.neutral_ready, self.stage = True, "ready"
            self.message = "Ready. Keep your shoulder and torso in the calibrated reference pose."
        self.calibration_version += 1
        self._samples = []
        self._capture_start = self._capture_pair = None
        self.progress = 1.

    def update(self, frame, now=None):
        now = _clock(now)
        if not isinstance(frame, dict) or frame.get("type", "frame") != "frame":
            return self.status(now)
        if self._last_clock is not None and now < self._last_clock:
            self.invalidate("Host clock moved backwards; repeat calibration.")
        source_generation = frame.get("generation")
        if source_generation is None:
            source_generation = self._source_generation
        timestamp = frame.get("t_us")
        samples = frame.get("sensors", [])
        if not isinstance(samples, list) or isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            self._latest = {}
            self.message = "Frame is missing sensor data or a valid timestamp"
            return self.status(now)
        if ((self._source_generation is not None and source_generation is not None and source_generation != self._source_generation)
                or (self._last_frame_t is not None and timestamp < self._last_frame_t)):
            self.invalidate("Device clock or generation changed; repeat calibration.")
        self._source_generation, self._last_frame_t, self._last_clock = source_generation, timestamp, now
        if self.stage in ("table", "neutral") and self._samples:
            if any(not item["ok"] for identity, item in self._health(now).items() if identity in self.mapping.values()):
                self._clear_capture()
        latest = {}
        for raw in samples:
            try:
                sample = normalize_sensor(raw, timestamp)
            except (ValueError, TypeError, KeyError):
                continue
            identity = sample["id"]
            if identity in latest:
                latest[identity]["ok"] = False
                latest[identity]["reason"] = "Duplicate sensor ID in frame"
                continue
            # Keep selected timestamps across a missing frame: an old sample must
            # not become fresh merely because the sensor disappeared briefly.
            old = self._history.get(identity, self._latest.get(identity))
            if identity in self.mapping.values() and old is not None and sample["t_us"] < old["t_us"]:
                self.invalidate("Selected sensor clock moved backwards; repeat calibration.")
                self._source_generation, self._last_frame_t, self._last_clock = source_generation, timestamp, now
                old = None
            sample["_received_at"] = now
            sample["_advanced_at"] = now if old is None or sample["t_us"] > old["t_us"] else old["_advanced_at"]
            latest[identity] = sample
        self._latest = latest
        self._history.update({identity: sample for identity, sample in latest.items() if identity in self.mapping.values()})
        if self.stage in ("table", "neutral"):
            try:
                self._capture(self._selected(now), now)
            except (ValueError, TypeError, KeyError) as error:
                self._clear_capture()
                self.message = str(error)
        return self.status(now)

    def status(self, now=None):
        now = _clock(now)
        health = self._health(now)
        selected_ok = all(health[identity]["ok"] for identity in self.mapping.values())
        stationary = False
        if selected_ok:
            try:
                stationary = self._stationarity(self._selected(now))
            except ValueError:
                pass
        if self.stage in ("table", "neutral") and not selected_ok:
            self._clear_capture()
            self.message = "Fresh samples from both selected sensors are required; capture restarted."
        duration = TABLE_SECONDS if self.stage == "table" else NEUTRAL_SECONDS if self.stage == "neutral" else 0.
        missing_motion = selected_ok and any(not health[identity]["has_gyro"] or not health[identity]["has_accel"] for identity in self.mapping.values())
        return {"stage": self.stage, "progress": float(self.progress), "message": self.message,
                "table_ready": bool(self.table_ready), "neutral_ready": bool(self.neutral_ready),
                "countdown_s": float(max(0., duration * (1 - self.progress))), "stationarity_ok": bool(stationary),
                "stationarity_note": ("Waiting for fresh samples from both selected sensors." if not selected_ok else
                                       "Missing gyro or acceleration: capture also relies on quaternion stability; hold the sensors still." if missing_motion else
                                       "Gyro, acceleration and quaternion stability checks are available."),
                "calibration_version": self.calibration_version, "generation": self.generation,
                "mapping": dict(self.mapping), "lengths_m": dict(self.lengths),
                "available_ids": sorted(self._latest), "sensors": health,
                "capture_samples": len(self._samples), "minimum_capture_samples": MIN_CAPTURE_SAMPLES}

    def pose(self, now=None):
        now = _clock(now)
        status = self.status(now)
        result = {"valid": False, "reason": "Complete table and mounted-neutral calibration first.",
                  "points": None, "segment_quaternions": None, "elbow_flexion_deg": None,
                  "coordinate_frame": "fixed shoulder: +X forward, +Y left, +Z up; metres",
                  "calibration_version": self.calibration_version, "generation": self.generation}
        try:
            selected = self._selected(now)
        except ValueError as error:
            result["reason"] = str(error)
            return result
        if not status["neutral_ready"]:
            return result
        qs = {role: multiply(multiply(self.alignment[role], sample["q"]), self.offset[role]) for role, sample in selected.items()}
        upper, forearm = (rotate(qs[role], (0, 0, -1)) for role in ("upper", "forearm"))
        elbow = upper * self.lengths["upper"]
        wrist = elbow + forearm * self.lengths["forearm"]
        result.update(valid=True, reason="", points={"shoulder": [0., 0., 0.], "elbow": elbow.tolist(), "wrist": wrist.tolist()},
                      segment_quaternions={role: q.tolist() for role, q in qs.items()},
                      elbow_flexion_deg=math.degrees(math.acos(float(np.clip(np.dot(upper, forearm), -1, 1)))))
        return result

    def snapshot(self):
        """JSON-safe calibration evidence for recordings; not automatic restore."""
        return deepcopy({"schema_version": 1, "mapping": self.mapping, "lengths_m": self.lengths,
                         "alignment_wxyz": {role: q.tolist() for role, q in self.alignment.items()},
                         "mounting_offset_wxyz": {role: q.tolist() for role, q in self.offset.items()},
                         "stage": self.stage, "table_ready": self.table_ready, "neutral_ready": self.neutral_ready,
                         "calibration_version": self.calibration_version, "generation": self.generation,
                         "device_generation": self._source_generation,
                         "frame": "fixed shoulder; +X forward, +Y left, +Z up; neutral -Z",
                         "method": "Orientation plus limb lengths; no acceleration integration or torso tracking."})
