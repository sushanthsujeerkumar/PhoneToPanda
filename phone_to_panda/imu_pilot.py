"""Prepare one explicitly experimental wrist path from a paired arm recording.

The IMUs supply orientation-derived wrist estimates relative to a fixed shoulder.
The video supplies manually reviewed grasp/release times. This is not a fused
tracker, a measured object trajectory, or a calibration-accuracy evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_trajectory(video, session, events, output):
    """Retain the recorded wrist estimates and explicit, provisional event times."""
    video, session, events, output = map(lambda p: Path(p).expanduser().resolve(),
                                        (video, session, events, output))
    if output.exists():
        raise FileExistsError(f"Preserve the existing trajectory: {output}")
    if not all(p.is_file() for p in (video, session, events)):
        raise FileNotFoundError("The original paired video, sensor session and event file are required.")
    recording = json.loads(session.read_text())
    labels = json.loads(events.read_text())
    if (labels.get("schema_version") != 1
            or labels.get("annotation_method") != "manual_video_review"
            or labels.get("synchronization") != "provisional"):
        raise ValueError("Use explicit manual video labels and provisional synchronization.")
    meta = recording["metadata"]
    if ("sensor_session_id" in labels and labels["sensor_session_id"] != meta.get("id")):
        raise ValueError("Manual events identify a different sensor session.")
    if "video_file" in labels and labels["video_file"] != video.name:
        raise ValueError("Manual events identify a different video file.")
    calibration = meta.get("calibration", {})
    if (meta.get("source") != "serial" or meta.get("synthetic") is not False
            or calibration.get("stage") != "ready"
            or not calibration.get("table_ready") or not calibration.get("neutral_ready")):
        raise ValueError("Use a real serial recording with completed table and neutral calibration.")
    mapping = calibration.get("mapping", {})
    if mapping != {"upper": "3", "forearm": "4"}:
        raise ValueError("This right-arm pilot expects upper sensor 3 and forearm sensor 4.")
    frames = recording.get("frames", [])
    if len(frames) < 4:
        raise ValueError("At least four recorded poses are required.")
    times = np.array([f["host_elapsed_s"] for f in frames], dtype=float)
    xyz = np.array([f["pose"]["points"]["wrist"] for f in frames], dtype=float)
    if (xyz.shape != (len(times), 3) or not np.isfinite(xyz).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or max(np.diff(times)) > .05):
        raise ValueError("Wrist samples must be finite, monotonic, and free of gaps over 50 ms.")
    for frame in frames:
        pose = frame["pose"]
        sensors = {str(s["id"]): s for s in frame["device_frame"]["sensors"]}
        if (pose.get("valid") is not True
                or pose.get("calibration_version") != calibration.get("calibration_version")
                or any(sensors.get(key, {}).get("ok") is not True for key in ("3", "4"))):
            raise ValueError("Use a consistently calibrated take with valid selected sensors and poses.")
    offset = float(labels["imu_time_offset_s"])
    video_grasp = float(labels["video_grasp_time_s"])
    video_release = float(labels["video_release_time_s"])
    if not np.isfinite([offset, video_grasp, video_release]).all() or not 0 <= video_grasp < video_release:
        raise ValueError("Video event times and the provisional offset must be finite and ordered.")
    grasp, release = video_grasp + offset, video_release + offset
    if not times[0] <= grasp < release <= times[-1]:
        raise ValueError("Aligned grasp/release times must lie inside the sensor recording.")
    trajectory = {
        "schema_version": 1,
        "coordinate_frame": "fixed_shoulder",
        "source": {
            "kind": "personal_video_imu_pilot", "video": str(video), "sha256": digest(video),
            "imu_session": str(session), "imu_sha256": digest(session),
            "events": str(events), "events_sha256": digest(events),
            "session_id": meta["id"], "selected_sensor_ids": ["3", "4"],
        },
        "grasp_time_s": grasp, "release_time_s": release,
        "samples": [{"t": float(t), "xyz": point.tolist(), "origin": "recorded_imu_wrist_estimate"}
                    for t, point in zip(times, xyz)],
        "quality": {
            "accepted": True, "acceptance_scope": "Acquisition and finite-path checks only; physical accuracy unmeasured.",
            "experimental": True, "estimated_wrist": True, "synchronization": "provisional",
            "imu_time_offset_s": offset, "camera_imu_extrinsics_calibrated": False,
            "camera_position_fusion": False, "source_recordings": 1,
            "configured_arm_lengths_m": calibration["lengths_m"],
            "calibration_version": calibration["calibration_version"],
            "valid_pose_frames": len(frames), "max_sample_gap_s": float(max(np.diff(times))),
            "method": "Recorded fixed-shoulder wrist estimates, manual video event labels, provisional clock offset. No gap filling or fabricated marker positions.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trajectory, indent=2, allow_nan=False) + "\n")
    return {"trajectory": str(output), "samples": len(times), "grasp_time_s": grasp,
            "release_time_s": release, "source": trajectory["source"], "quality": trajectory["quality"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_trajectory(args.video, args.session, args.events, args.output), indent=2))


if __name__ == "__main__":
    main()
