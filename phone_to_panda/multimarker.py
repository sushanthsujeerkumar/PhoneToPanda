"""Fixed-camera tracking with a fitted coplanar reference-marker board.

Reference locations are fitted from detected corners and measured marker sizes.
Their spacing need not form a grid. All points remain in marker 0's frame. Image
regions separate the existing recordings' object and reference markers with ID 1.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import json

import numpy as np

from .video import (_opencv, _open_video, _portable_source_path, _save_json, _sha256,
                    ARUCO_DICTIONARY, marker_object_points, estimate_marker_pose,
                    validate_events, interpolate_short_gaps)


def detections_from_opencv(corners, ids):
    return [] if ids is None else [
        {"id": int(key), "corners": np.asarray(corner).reshape(4, 2).tolist()}
        for key, corner in zip(ids.ravel(), corners)]


def reference_detections(detections, reference_markers, image_size):
    """Select by ID AND an explicit image region; never collapse duplicate IDs."""
    scale = np.asarray(image_size, dtype=float)
    selected = {}
    for spec in reference_markers:
        marker_id = int(spec["id"])
        region = np.asarray(spec["region"], dtype=float)
        if region.shape != (4,) or not np.isfinite(region).all() or np.any(region < 0) or np.any(region > 1) or np.any(region[2:] <= region[:2]):
            raise ValueError("Reference regions must be normalized [xmin, ymin, xmax, ymax].")
        candidates = []
        for detection in detections:
            corners = np.asarray(detection["corners"], dtype=float).reshape(4, 2)
            center = corners.mean(0) / scale
            if int(detection["id"]) == marker_id and np.all(center >= region[:2]) and np.all(center <= region[2:]):
                candidates.append(corners)
        if len(candidates) > 1:
            raise ValueError(f"Reference region for ID {marker_id} contains duplicate candidates; narrow its region.")
        if candidates:
            selected[marker_id] = candidates[0]
    return selected


def _pose_solutions(corners, size, matrix, distortion):
    cv2 = _opencv()
    points = marker_object_points(size)
    answer = cv2.solvePnPGeneric(points, corners, matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    poses = []
    for rvec, tvec in zip(answer[1], answer[2]):
        rotation = cv2.Rodrigues(rvec)[0]
        translation = np.asarray(tvec).reshape(3)
        if np.isfinite(rotation).all() and np.isfinite(translation).all() and np.all(((rotation @ points.T).T + translation)[:, 2] > 0):
            poses.append((rotation, translation))
    if not poses:
        raise ValueError("Reference marker pose could not be estimated in front of camera.")
    return poses


def fit_reference_board(frames, reference_markers, image_size, marker_size_m,
                        matrix, distortion, max_reprojection_error_px=3.0):
    """Fit one flat reference surface and each marker's position and rotation.

    A square can give two camera-pose solutions. Start with the combination whose
    surface normals agree most closely, then jointly fit all detected corners.
    Marker 0 fixes the origin and axes; measured marker size sets the metric scale.
    The reprojection-error limit rejects a poor fit but is not a ground-truth check.
    """
    from scipy.optimize import least_squares
    cv2 = _opencv()
    ids = [int(spec["id"]) for spec in reference_markers]
    if len(ids) < 3 or len(ids) > 8 or len(set(ids)) != len(ids) or 0 not in ids:
        raise ValueError("Use 3–8 unique reference IDs including anchor ID 0.")
    ids = [0] + sorted(key for key in ids if key != 0)
    observations = {key: [] for key in ids}
    for frame in frames:
        for key, corners in reference_detections(frame["detections"], reference_markers, image_size).items():
            observations[key].append(corners)
    if any(len(observations[key]) < 3 for key in ids):
        raise ValueError("Each reference must be visible in at least three initial frames to fit its common plane.")
    measured = {key: np.median(observations[key], axis=0) for key in ids}
    poses = {key: _pose_solutions(measured[key], marker_size_m, matrix, distortion) for key in ids}
    choices = itertools.product(*(range(len(poses[key])) for key in ids))
    def normal_variance(choice):
        normals = np.array([poses[key][index][0][:, 2] for key, index in zip(ids, choice)])
        return float(np.sum((normals - normals.mean(0)) ** 2))
    choice = min(choices, key=normal_variance)
    chosen = [poses[key][index] for key, index in zip(ids, choice)]
    anchor_r, anchor_t = chosen[0]
    initial = list(cv2.Rodrigues(anchor_r)[0].ravel()) + list(anchor_t)
    for rotation, translation in chosen[1:]:
        local_t = anchor_r.T @ (translation - anchor_t)
        local_r = anchor_r.T @ rotation
        initial.extend([*local_t[:2], math.atan2(local_r[1, 0], local_r[0, 0])])
    square = marker_object_points(marker_size_m)
    def points(params):
        parts = [square]
        for index in range(len(ids) - 1):
            x, y, angle = params[6 + 3 * index:9 + 3 * index]
            c, s = math.cos(angle), math.sin(angle)
            rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            parts.append((rz @ square.T).T + [x, y, 0])
        return np.concatenate(parts)
    measured_array = np.concatenate([measured[key] for key in ids])
    def residual(params):
        projected = cv2.projectPoints(points(params), params[:3], params[3:6], matrix, distortion)[0]
        return (projected.reshape(-1, 2) - measured_array).ravel()
    fit = least_squares(residual, initial, loss="soft_l1", f_scale=1.0,
                        max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    rotation = cv2.Rodrigues(fit.x[:3])[0]
    translation = fit.x[3:6]
    board_points = points(fit.x)
    errors = np.linalg.norm(residual(fit.x).reshape(len(ids), 4, 2), axis=2)
    per_marker_rms = np.sqrt(np.mean(errors ** 2, axis=1))
    if not fit.success or not np.isfinite(fit.x).all() or np.any(((rotation @ board_points.T).T + translation)[:, 2] <= 0):
        raise ValueError("Common reference-plane fit failed or fell behind camera.")
    if float(max(per_marker_rms)) > max_reprojection_error_px:
        raise ValueError(f"Reference squares are inconsistent with one plane: worst marker RMS {max(per_marker_rms):.2f}px exceeds {max_reprojection_error_px:.2f}px. Check flat placement, printed black-square sizes, and calibration.")
    return {"schema_version": 1, "coordinate_frame": "table_marker", "anchor_id": 0,
            "marker_size_m": float(marker_size_m), "reference_markers": reference_markers,
            "camera_rotation": rotation.tolist(), "camera_translation": translation.tolist(),
            "reference_image_corners": {str(key): measured[key].tolist() for key in ids},
            "reference_board_corners_m": {str(key): part.tolist() for key, part in zip(ids, board_points.reshape(-1, 4, 3))},
            "per_marker_rms_px": {str(key): float(error) for key, error in zip(ids, per_marker_rms)},
            "rms_reprojection_error_px": float(np.sqrt(np.mean(errors ** 2))),
            "initial_observation_counts": {str(key): len(observations[key]) for key in ids},
            "method": "Joint coplanar equal-size square fit, with free in-plane marker positions and yaws; fixed camera; anchor ID 0."}


def select_object_detection(detections, board, object_marker_id=1, exclusion_margin_px=15):
    """Remove candidates inside fixed reference footprints, including duplicate ID."""
    candidates = []
    for detection in detections:
        if int(detection["id"]) != object_marker_id:
            continue
        corners = np.asarray(detection["corners"], float).reshape(4, 2)
        center = corners.mean(0)
        overlaps_reference = False
        for reference_corners in board["reference_image_corners"].values():
            reference = np.asarray(reference_corners)
            if np.all(center >= reference.min(0) - exclusion_margin_px) and np.all(center <= reference.max(0) + exclusion_margin_px):
                overlaps_reference = True
                break
        if not overlaps_reference:
            candidates.append(corners)
    # An occluded object is missing even if the stationary ID 1 remains visible.
    return candidates[0] if len(candidates) == 1 else None


def extract_video_multimarker(video_path, calibration_path, events_path, output_path,
                              reference_markers, table_marker_size_m=0.05, object_marker_size_m=0.03,
                              object_marker_id=1, sample_hz=30, max_gap_s=0.2,
                              min_valid_fraction=0.90, max_pose_reprojection_error_px=3.0,
                              max_table_drift_px=10.0, resting_height_range_m=(-0.01, 0.15),
                              preview_path=None, quality_path=None, write_preview=False):
    """Track the object-top marker centre in the fixed reference frame, in metres.

    Quality checks cover calibration, reference drift, image reprojection error,
    and resting marker height. Measured coverage is counted before interpolation;
    only short gaps bounded by observed samples are filled afterward.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Trajectory output already exists: {output_path}. Choose a new output.")
    if sample_hz <= 0 or not 0 < min_valid_fraction <= 1:
        raise ValueError("sample_hz must be positive and min_valid_fraction in (0,1].")
    marker_object_points(table_marker_size_m)
    marker_object_points(object_marker_size_m)
    bounds = np.asarray(resting_height_range_m, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError("resting_height_range_m must contain [minimum, maximum] metres.")
    calibration = json.loads(Path(calibration_path).read_text())
    if calibration.get("accepted") is not True:
        raise ValueError("Use an accepted calibration produced by calibrate-camera.")
    matrix = np.asarray(calibration["camera_matrix"], float)
    distortion = np.asarray(calibration["distortion_coefficients"], float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Invalid calibration matrix or distortion values.")
    quality_path = Path(quality_path) if quality_path else output_path.with_suffix(".quality.json")
    board_path = output_path.with_suffix(".board.json")
    cv2 = _opencv()
    capture, metadata = _open_video(video_path)
    quality = {"accepted": False, "tracking_mode": "multi_marker", "video_metadata": metadata,
               "coordinate_note": "Marker 0 printed axes; +z above common reference plane; object-top marker center in metres."}
    writer = None
    try:
        if metadata["image_size"] != calibration["image_size"]:
            raise ValueError("Recording resolution differs from camera calibration.")
        events = validate_events(json.loads(Path(events_path).read_text()), metadata["duration_s"])
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTIONARY))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        initial_frames = []
        for _ in range(max(3, int(metadata["fps"] * 0.5))):
            ok, frame = capture.read()
            if not ok:
                break
            corners, ids, _ = detector.detectMarkers(frame)
            initial_frames.append({"detections": detections_from_opencv(corners, ids)})
        board = fit_reference_board(initial_frames, reference_markers, metadata["image_size"],
                                    table_marker_size_m, matrix, distortion, max_pose_reprojection_error_px)
        board.update({"source_video_sha256": _sha256(video_path),
                      "calibration_sha256": _sha256(calibration_path)})
        _save_json(board_path, board)
        quality["reference_board"] = _portable_source_path(board_path)
        quality["reference_board_rms_px"] = board["rms_reprojection_error_px"]
        capture.release()
        capture, _ = _open_video(video_path)
        rotation = np.asarray(board["camera_rotation"])
        translation = np.asarray(board["camera_translation"])
        stride = max(1, round(metadata["fps"] / sample_hz))
        actual_hz = metadata["fps"] / stride
        first = max(0, int(math.floor(events["start_time_s"] * metadata["fps"] / stride)) * stride)
        last = min(int(math.ceil(events["end_time_s"] * metadata["fps"] / stride)) * stride, metadata["frame_count"] - 1)
        if write_preview:
            preview_path = Path(preview_path) if preview_path else output_path.with_suffix(".preview.mp4")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), actual_hz, tuple(metadata["image_size"]))
            if not writer.isOpened():
                raise ValueError(f"Could not create preview: {preview_path}")
        positions, times, errors = [], [], []
        drift = {str(spec["id"]): 0.0 for spec in reference_markers}
        missing_object = 0
        read_frames = 0
        for frame_index in range(last + 1):
            ok, frame = capture.read()
            if not ok:
                break
            read_frames += 1
            if frame_index < first or not (frame_index % stride == 0 or frame_index == last):
                continue
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(timestamp) or (frame_index > 0 and timestamp <= 0) or (times and timestamp <= times[-1]):
                timestamp = frame_index / metadata["fps"]
            corners, ids, _ = detector.detectMarkers(frame)
            detections = detections_from_opencv(corners, ids)
            references = reference_detections(detections, reference_markers, metadata["image_size"])
            for key, measured in references.items():
                distance = np.linalg.norm(measured - np.asarray(board["reference_image_corners"][str(key)]), axis=1)
                drift[str(key)] = max(drift[str(key)], float(max(distance)))
            object_corners = select_object_detection(detections, board, object_marker_id)
            position = [float("nan")] * 3
            note = "MISSING OR AMBIGUOUS OBJECT"
            # At least one currently visible reference verifies the fixed frame.
            if object_corners is not None and references:
                try:
                    _, object_translation, error = estimate_marker_pose(object_corners, object_marker_size_m, matrix, distortion)
                    errors.append(error)
                    if error <= max_pose_reprojection_error_px:
                        position = (rotation.T @ (object_translation - translation)).tolist()
                        note = "xyz m: " + ", ".join(f"{value:+.3f}" for value in position)
                except (ValueError, cv2.error):
                    pass
            else:
                missing_object += 1
            positions.append(position)
            times.append(timestamp)
            if writer:
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.putText(frame, f"t={timestamp:.2f}s {note}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 200, 0) if np.isfinite(position).all() else (0, 0, 255), 2)
                writer.write(frame)
        positions = np.asarray(positions)
        times = np.asarray(times)
        valid = np.isfinite(positions).all(axis=1)
        fraction = float(valid.mean()) if len(valid) else 0.0
        quality.update({"sample_count": len(times), "measured_samples": int(valid.sum()),
                        "measured_fraction": fraction, "sample_hz": actual_hz, "max_allowed_gap_s": max_gap_s,
                        "max_table_corner_drift_px": max(drift.values()), "per_reference_drift_px": drift,
                        "max_pose_reprojection_error_px": max(errors) if errors else None,
                        "median_pose_reprojection_error_px": float(np.median(errors)) if errors else None,
                        "missing_or_ambiguous_object_samples": missing_object,
                        "preview": _portable_source_path(preview_path) if writer else None})
        if read_frames < last + 1:
            raise ValueError("Video decoding ended before annotated end.")
        if fraction < min_valid_fraction:
            raise ValueError(f"Only {fraction:.1%} measured samples; need {min_valid_fraction:.0%}.")
        if max(drift.values()) > max_table_drift_px:
            raise ValueError(f"A reference moved by {max(drift.values()):.1f}px; limit is {max_table_drift_px:.1f}px. Keep camera and all references fixed.")
        rest = positions[valid & (times < events["grasp_time_s"]), 2]
        if len(rest) < 3:
            raise ValueError("Need at least three measured pre-grasp samples to verify resting marker height.")
        resting_height = float(np.median(rest))
        quality["resting_height_check"] = {"median_marker_top_height_m": resting_height,
                                           "allowed_range_m": bounds.tolist(), "samples": len(rest)}
        if not bounds[0] <= resting_height <= bounds[1]:
            raise ValueError(f"Resting object marker height {resting_height:.3f}m is outside {bounds.tolist()}m. Measure BLACK marker edges and verify calibration; geometry cannot be corrected by subtracting a guessed height.")
        filled, gaps = interpolate_short_gaps(times, positions, max_gap_s)
        if times[0] > events["start_time_s"] + 1 / metadata["fps"] or times[-1] < events["end_time_s"] - 1 / metadata["fps"]:
            raise ValueError("Decoded timestamps do not cover annotation.")
        inside = (times > events["start_time_s"]) & (times < events["end_time_s"])
        output_times = np.r_[events["start_time_s"], times[inside], events["end_time_s"]]
        output_positions = np.column_stack([np.interp(output_times, times, filled[:, dim]) for dim in range(3)])
        target = np.array([np.interp(events["release_time_s"], times, filled[:, dim]) for dim in range(3)])
        quality.update({"accepted": True, "interpolated_gaps": gaps,
                        "interpolated_samples": sum(gap["filled_samples"] for gap in gaps),
                        "trajectory_extent_m": np.ptp(output_positions, axis=0).tolist()})
        result = {"schema_version": 1,
                  "source": {"video": _portable_source_path(video_path), "sha256": _sha256(video_path),
                             "kind": "personal_video", "calibration_sha256": _sha256(calibration_path),
                             "events_sha256": _sha256(events_path), "reference_board_sha256": _sha256(board_path)},
                  "coordinate_frame": "table_marker", "units": "metres",
                  "samples": [{"t": float(t), "xyz": xyz.tolist()} for t, xyz in zip(output_times, output_positions)],
                  "grasp_time_s": events["grasp_time_s"], "release_time_s": events["release_time_s"],
                  "target_xyz": target.tolist(), "markers": {"dictionary": ARUCO_DICTIONARY, "table_id": 0,
                  "object_id": object_marker_id, "table_marker_size_m": table_marker_size_m,
                  "object_marker_size_m": object_marker_size_m, "reference_markers": reference_markers}, "quality": quality}
        _save_json(quality_path, quality)
        _save_json(output_path, result)
        return result
    except (ValueError, FileNotFoundError) as error:
        quality["rejection_reason"] = str(error)
        _save_json(quality_path, quality)
        raise
    finally:
        capture.release()
        if writer:
            writer.release()
