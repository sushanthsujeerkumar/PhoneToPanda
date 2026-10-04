"""Calibrated phone video -> measured object trajectory.

No robot demonstrations are invented here. A marker on the object supplies its
measured position; grasp/release times are explicitly annotated by the user.
OpenCV is imported lazily so help and configuration work before installation.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np


ARUCO_DICTIONARY = "DICT_4X4_50"


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the video requirements first (opencv-contrib-python-headless).") from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV needs the contrib package for ArUco marker support.")
    return cv2


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_source_path(path):
    """Keep recordings portable when the entire project folder is moved."""
    resolved = Path(path).resolve()
    project_root = Path(__file__).resolve().parents[1]
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def _open_video(path):
    cv2 = _opencv()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    if not math.isfinite(fps) or fps <= 0 or count < 2 or min(size) <= 0:
        capture.release()
        raise ValueError("Video needs readable frame count, frame rate, and dimensions. Export an MP4 first.")
    return capture, {"fps": fps, "frame_count": count, "image_size": size, "duration_s": (count - 1) / fps}


def _png_with_dpi(path, pixels, dpi=300):
    """Write PNG physical-resolution metadata so printing at 100% has a scale."""
    cv2 = _opencv()
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise RuntimeError("Could not encode calibration PNG.")
    raw = encoded.tobytes()
    ppm = round(dpi / 0.0254)
    payload = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    # PNG signature (8 bytes), IHDR chunk (25 bytes), then physical resolution.
    Path(path).write_bytes(raw[:33] + chunk + raw[33:])


def generate_calibration_assets(output_dir, table_marker_mm=50.0, object_marker_mm=30.0,
                                chess_square_mm=25.0, board_inner_corners=(9, 6)):
    """Create four table references, an object marker, and a calibration board."""
    cv2 = _opencv()
    if min(table_marker_mm, object_marker_mm, chess_square_mm) <= 0:
        raise ValueError("Printed dimensions must be positive.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = 300
    px_per_mm = dpi / 25.4
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTIONARY))
    files = {}
    markers = [(i, table_marker_mm, f"table_marker_{i}") for i in range(4)]
    markers.append((1, object_marker_mm, "object_marker_1"))
    for marker_id, size_mm, name in markers:
        width = round(size_mm * px_per_mm)
        # A 30 mm object marker plus 5 mm borders fits a 40 mm cube top.
        margin = round((5 if name == "object_marker_1" else 10) * px_per_mm)
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, width)
        page = np.full((width + 2 * margin, width + 2 * margin), 255, dtype=np.uint8)
        page[margin:margin + width, margin:margin + width] = marker
        path = output_dir / f"{name}.png"
        _png_with_dpi(path, page, dpi)
        files[name] = str(path)
    cols, rows = (int(x) + 1 for x in board_inner_corners)
    if cols < 4 or rows < 4:
        raise ValueError("Chessboard needs at least three inner corners in both directions.")
    square = round(chess_square_mm * px_per_mm)
    margin = round(10 * px_per_mm)
    board = np.full((rows * square + 2 * margin, cols * square + 2 * margin), 255, np.uint8)
    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 0:
                board[margin + y * square:margin + (y + 1) * square,
                      margin + x * square:margin + (x + 1) * square] = 0
    path = output_dir / "chessboard.png"
    _png_with_dpi(path, board, dpi)
    files["chessboard"] = str(path)
    manifest = {"schema_version": 1, "dictionary": ARUCO_DICTIONARY, "dpi": dpi,
                "table_marker_ids": [0, 1, 2, 3], "table_marker_size_m": table_marker_mm / 1000,
                "object_marker_id": 1, "object_marker_size_m": object_marker_mm / 1000,
                "chessboard_inner_corners": list(board_inner_corners),
                "chessboard_square_size_m": chess_square_mm / 1000, "files": files,
                "printing": "Print at actual size / 100%, not fit-to-page. Measure the black marker square and chessboard squares with a ruler; use measured dimensions in configuration."}
    _save_json(output_dir / "dimensions.json", manifest)
    return manifest


def calibrate_camera(video_path, output_path, board_inner_corners=(9, 6), square_size_m=0.025,
                     min_views=10, sample_every_s=0.4, max_reprojection_error_px=1.5):
    """Estimate pinhole intrinsics from a separate video of a moved/tilted chessboard."""
    cv2 = _opencv()
    if square_size_m <= 0 or min_views < 10 or sample_every_s <= 0:
        raise ValueError("Use a positive square size and sampling interval, and at least 10 calibration views.")
    board_inner_corners = tuple(int(v) for v in board_inner_corners)
    if len(board_inner_corners) != 2 or min(board_inner_corners) < 3:
        raise ValueError("Chessboard inner-corner dimensions must both be at least 3.")
    capture, metadata = _open_video(video_path)
    stride = max(1, round(metadata["fps"] * sample_every_s))
    grid = np.zeros((math.prod(board_inner_corners), 3), np.float32)
    grid[:, :2] = np.mgrid[0:board_inner_corners[0], 0:board_inner_corners[1]].T.reshape(-1, 2) * square_size_m
    observations = []
    descriptors = []
    sampled = 0
    frame_index = 0
    diagonal = float(np.linalg.norm(metadata["image_size"]))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                sampled += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCornersSB(gray, board_inner_corners)
                if found:
                    flat = corners.reshape(-1, 2)
                    # Reject near-identical views: duplicate frames do not add calibration evidence.
                    if not descriptors or min(np.sqrt(np.mean(np.sum((flat - prev) ** 2, axis=1)))
                                              for prev in descriptors) > 0.025 * diagonal:
                        observations.append(corners.astype(np.float32))
                        descriptors.append(flat)
            frame_index += 1
            if len(observations) >= 60:
                break
    finally:
        capture.release()
    if len(observations) < min_views:
        raise ValueError(f"Only {len(observations)} distinct chessboard views found; need {min_views}. Slowly tilt and move the entire board around the image, keeping all corners visible.")
    # Flat, front-facing boards translated around the image are still degenerate.
    # Foreshortening/shear in the board's two image-plane edge vectors establishes
    # that at least some views were tilted, rather than merely translated.
    perspective = []
    cols, rows = board_inner_corners
    for corners in descriptors:
        horizontal = corners[cols - 1] - corners[0]
        vertical = corners[(rows - 1) * cols] - corners[0]
        horizontal2 = corners[-1] - corners[(rows - 1) * cols]
        vertical2 = corners[-1] - corners[cols - 1]
        shear = abs(float(np.dot(horizontal, vertical))) / max(float(np.linalg.norm(horizontal) * np.linalg.norm(vertical)), 1e-6)
        convergence = max(abs(np.linalg.norm(horizontal) / max(np.linalg.norm(horizontal2), 1e-6) - 1),
                          abs(np.linalg.norm(vertical) / max(np.linalg.norm(vertical2), 1e-6) - 1))
        perspective.append(max(shear, convergence))
    if max(perspective) < 0.07:
        raise ValueError("Calibration views look almost front-facing. Record the board tilted in several directions, not just moving sideways.")
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        [grid.copy() for _ in observations], observations, tuple(metadata["image_size"]), None, None)
    errors = []
    for measured, rvec, tvec in zip(observations, rvecs, tvecs):
        projected, _ = cv2.projectPoints(grid, rvec, tvec, matrix, distortion)
        errors.append(float(np.sqrt(np.mean(np.sum((measured.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2, axis=1)))))
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all() or not math.isfinite(rms) or not np.isfinite(errors).all():
        raise ValueError("Calibration produced non-finite values. Re-record sharp, varied board poses; no calibration was written.")
    report = {"schema_version": 1, "camera_matrix": matrix.tolist(),
              "distortion_coefficients": distortion.ravel().tolist(), "image_size": metadata["image_size"],
              "board_inner_corners": list(board_inner_corners), "square_size_m": square_size_m,
              "views_used": len(observations), "frames_sampled": sampled,
              "rms_reprojection_error_px": float(rms), "per_view_error_px": errors,
              "source_video": _portable_source_path(video_path), "source_sha256": _sha256(video_path)}
    if rms > max_reprojection_error_px:
        report["accepted"] = False
        _save_json(Path(output_path).with_suffix(".rejected.json"), report)
        raise ValueError(f"Calibration RMS {rms:.2f}px exceeds {max_reprojection_error_px:.2f}px. Re-record sharper, varied views. A rejected report was saved; no usable calibration was written.")
    report["accepted"] = True
    _save_json(output_path, report)
    return report


def write_events_template(video_path, output_path):
    capture, metadata = _open_video(video_path)
    capture.release()
    template = {"schema_version": 1, "start_time_s": 0.0, "grasp_time_s": None,
                "release_time_s": None, "end_time_s": None,
                "video_duration_s": metadata["duration_s"],
                "notes": "Fill grasp_time_s when fingers close on the object, and release_time_s when they open over the box. Times are seconds from video start. Null end_time_s uses release_time_s; choose an end where both markers remain visible."}
    _save_json(output_path, template)
    return template


def validate_events(events, duration_s):
    def number(name, default=None):
        value = events.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Set {name} to a finite number of seconds in the events JSON.")
        return float(value)
    grasp = number("grasp_time_s")
    release = number("release_time_s")
    start = number("start_time_s", 0.0)
    end = release if events.get("end_time_s") is None else number("end_time_s")
    if not (0 <= start < grasp < release <= end <= duration_s + 1e-3):
        raise ValueError(f"Event times must satisfy 0 <= start < grasp < release <= end <= video duration ({duration_s:.3f}s).")
    return {"start_time_s": start, "grasp_time_s": grasp, "release_time_s": release, "end_time_s": end}


def marker_object_points(marker_size_m):
    """IPPE_SQUARE order: top left, top right, bottom right, bottom left."""
    half = float(marker_size_m) / 2
    if not math.isfinite(half) or half <= 0:
        raise ValueError("Marker size must be positive metres.")
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], dtype=np.float64)


def estimate_marker_pose(corners, marker_size_m, camera_matrix, distortion):
    """Return marker-to-camera rotation, translation, and reprojection RMS."""
    cv2 = _opencv()
    object_points = marker_object_points(marker_size_m)
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    answer = cv2.solvePnPGeneric(object_points, image_points, camera_matrix, distortion,
                                 flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not answer[0]:
        raise ValueError("Marker pose could not be estimated.")
    solutions = []
    for rvec, tvec in zip(answer[1], answer[2]):
        rotation, _ = cv2.Rodrigues(rvec)
        translation = tvec.reshape(3)
        if not np.isfinite(translation).all() or np.any((rotation @ object_points.T).T[:, 2] + translation[2] <= 0):
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1))))
        if math.isfinite(error):
            solutions.append((error, rotation, translation))
    if not solutions:
        raise ValueError("Marker pose is behind the camera or non-finite.")
    error, rotation, translation = min(solutions, key=lambda item: item[0])
    return rotation, translation, error


def camera_point_in_table(point_camera, table_rotation, table_translation):
    """Transform a camera-space point to the fixed table marker's metric frame."""
    return np.asarray(table_rotation).T @ (np.asarray(point_camera).reshape(3) - np.asarray(table_translation).reshape(3))


def interpolate_short_gaps(times, positions, max_gap_s=0.2):
    """Interpolate only internally bounded gaps; never extrapolate hidden endpoints."""
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float).copy()
    if times.ndim != 1 or positions.shape != (len(times), 3) or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory needs at least two strictly increasing timestamps and matching xyz samples.")
    if not math.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError("max_gap_s must be finite and non-negative.")
    valid = np.isfinite(positions).all(axis=1)
    if not valid[0] or not valid[-1]:
        raise ValueError("A marker is missing at the start or end. Trim to visible endpoints; extrapolation is disabled.")
    indices = np.flatnonzero(valid)
    gaps = []
    for before, after in zip(indices[:-1], indices[1:]):
        if after == before + 1:
            continue
        span = float(times[after] - times[before])
        if span > max_gap_s + 1e-9:
            raise ValueError(f"Marker occlusion spans {span:.3f}s, exceeding {max_gap_s:.3f}s. Re-record with the markers visible or trim the clip.")
        alpha = (times[before + 1:after] - times[before]) / span
        positions[before + 1:after] = positions[before] + alpha[:, None] * (positions[after] - positions[before])
        gaps.append({"start_s": float(times[before]), "end_s": float(times[after]), "filled_samples": int(after - before - 1)})
    return positions, gaps


def extract_video(video_path, calibration_path, events_path, output_path, preview_path=None,
                  quality_path=None, table_marker_size_m=0.05, object_marker_size_m=0.03,
                  sample_hz=20.0, max_gap_s=0.2, min_valid_fraction=0.90,
                  max_pose_reprojection_error_px=3.0, max_table_drift_px=10.0,
                  tracking_mode="single_marker", reference_markers=None,
                  object_marker_id=1, write_preview=True,
                  resting_height_range_m=(-0.01, 0.15)):
    """Measure a marked object's path. Save provenance, diagnostic preview, and QA.

    Failed quality gates produce diagnostics and raise ValueError. They never
    create a trajectory advertised as ready for retargeting.
    """
    if tracking_mode == "multi_marker":
        from .multimarker import extract_video_multimarker
        return extract_video_multimarker(
            video_path, calibration_path, events_path, output_path,
            reference_markers=reference_markers or [], table_marker_size_m=table_marker_size_m,
            object_marker_size_m=object_marker_size_m, object_marker_id=object_marker_id,
            sample_hz=sample_hz, max_gap_s=max_gap_s, min_valid_fraction=min_valid_fraction,
            max_pose_reprojection_error_px=max_pose_reprojection_error_px,
            max_table_drift_px=max_table_drift_px, resting_height_range_m=resting_height_range_m,
            preview_path=preview_path, quality_path=quality_path, write_preview=write_preview)
    if tracking_mode != "single_marker":
        raise ValueError("tracking_mode must be single_marker or multi_marker.")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Trajectory output already exists: {output_path}. Choose --output with a new filename so a failed preparation cannot be mistaken for an earlier success.")
    cv2 = _opencv()
    if sample_hz <= 0 or not 0 < min_valid_fraction <= 1:
        raise ValueError("sample_hz must be positive; min_valid_fraction must be in (0, 1].")
    marker_object_points(table_marker_size_m)
    marker_object_points(object_marker_size_m)
    calibration = json.loads(Path(calibration_path).read_text())
    if calibration.get("accepted") is not True:
        raise ValueError("Use an accepted calibration produced by calibrate-camera.")
    matrix = np.asarray(calibration["camera_matrix"], np.float64)
    distortion = np.asarray(calibration["distortion_coefficients"], np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Invalid calibration matrix or distortion values.")
    capture, metadata = _open_video(video_path)
    try:
        if metadata["image_size"] != calibration["image_size"]:
            raise ValueError(f"Video resolution {metadata['image_size']} differs from calibration {calibration['image_size']}. Calibrate using the identical camera, lens, zoom, orientation, and resolution.")
        events = validate_events(json.loads(Path(events_path).read_text()), metadata["duration_s"])
    except Exception:
        capture.release()
        raise
    quality_path = Path(quality_path) if quality_path else output_path.with_suffix(".quality.json")
    preview_path = Path(preview_path) if preview_path else output_path.with_suffix(".preview.mp4")
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, round(metadata["fps"] / sample_hz))
    actual_hz = metadata["fps"] / stride
    writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), actual_hz,
                             tuple(metadata["image_size"]))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not create preview video at {preview_path}.")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTIONARY))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    positions, times, errors, table_corners = [], [], [], []
    frame_index = 0
    first = max(0, int(math.floor(events["start_time_s"] * metadata["fps"] / stride)) * stride)
    last = int(math.ceil(events["end_time_s"] * metadata["fps"] / stride)) * stride
    last = min(last, metadata["frame_count"] - 1)
    read_frames = 0
    try:
        while frame_index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            read_frames += 1
            use_frame = frame_index >= first and (frame_index % stride == 0 or frame_index == last)
            if use_frame:
                # Timestamp from the decoder, with frame-rate fallback for containers
                # that supply no presentation timestamps. No time-warp is applied.
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                if not math.isfinite(timestamp) or (frame_index > 0 and timestamp <= 0) or (times and timestamp <= times[-1]):
                    timestamp = frame_index / metadata["fps"]
                corners, ids, _ = detector.detectMarkers(frame)
                labels = {} if ids is None else {int(key): corner for key, corner in zip(ids.ravel(), corners)}
                position = [float("nan")] * 3
                note = "MISSING MARKER"
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                if 0 in labels:
                    table_corners.append(labels[0].reshape(4, 2).copy())
                if 0 in labels and 1 in labels:
                    try:
                        table_r, table_t, table_error = estimate_marker_pose(labels[0], table_marker_size_m, matrix, distortion)
                        _, object_t, object_error = estimate_marker_pose(labels[1], object_marker_size_m, matrix, distortion)
                        pose_error = max(table_error, object_error)
                        errors.append(pose_error)
                        if pose_error <= max_pose_reprojection_error_px:
                            position = camera_point_in_table(object_t, table_r, table_t).tolist()
                            note = "xyz m: " + ", ".join(f"{value:+.3f}" for value in position)
                            table_rvec, _ = cv2.Rodrigues(table_r)
                            cv2.drawFrameAxes(frame, matrix, distortion, table_rvec, table_t, table_marker_size_m)
                        else:
                            note = f"REJECTED POSE: {pose_error:.1f}px"
                    except (ValueError, cv2.error):
                        note = "POSE FAILED"
                times.append(timestamp)
                positions.append(position)
                cv2.putText(frame, f"t={timestamp:.2f}s  {note}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 200, 0) if np.isfinite(position).all() else (0, 0, 255), 2)
                writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    valid_count = sum(bool(np.isfinite(p).all()) for p in positions)
    fraction = valid_count / len(positions) if positions else 0
    table_drift = 0.0
    if table_corners:
        table_corners = np.asarray(table_corners)
        centre = np.median(table_corners, axis=0)
        table_drift = float(np.max(np.linalg.norm(table_corners - centre, axis=2)))
    quality = {"accepted": False, "sample_count": len(positions), "measured_samples": valid_count,
               "measured_fraction": fraction, "sample_hz": actual_hz,
               "max_allowed_gap_s": max_gap_s, "max_table_corner_drift_px": table_drift,
               "max_pose_reprojection_error_px": max(errors) if errors else None,
               "median_pose_reprojection_error_px": float(np.median(errors)) if errors else None,
               "preview": _portable_source_path(preview_path), "video_metadata": metadata,
               "coordinate_note": "+x toward printed right, +y toward printed top, +z out of the table marker plane; metres; position is marker centre on object top, not object centre."}
    try:
        if read_frames < last + 1:
            raise ValueError("Video decoding stopped before the annotated end. Re-export the clip as a readable MP4.")
        if fraction < min_valid_fraction:
            raise ValueError(f"Only {fraction:.1%} of samples have both usable markers; need {min_valid_fraction:.0%}. See the preview and re-record without occlusion.")
        if table_drift > max_table_drift_px:
            raise ValueError(f"Table marker moved by up to {table_drift:.1f}px. Keep the phone and table marker fixed; disable video stabilisation.")
        filled, gaps = interpolate_short_gaps(times, positions, max_gap_s)
        times = np.asarray(times)
        if times[0] > events["start_time_s"] + 1 / metadata["fps"] or times[-1] < events["end_time_s"] - 1 / metadata["fps"]:
            raise ValueError("Decoded timestamps do not cover the annotation. Export a constant-frame-rate MP4 and annotate that exported file.")
        # Exact clip endpoints avoid passing unrelated motion to the robot.
        inside = (times > events["start_time_s"]) & (times < events["end_time_s"])
        output_times = np.r_[events["start_time_s"], times[inside], events["end_time_s"]]
        output_positions = np.column_stack([np.interp(output_times, times, filled[:, dim]) for dim in range(3)])
        target = np.array([np.interp(events["release_time_s"], times, filled[:, dim]) for dim in range(3)])
        quality.update({"accepted": True, "interpolated_gaps": gaps,
                        "interpolated_samples": sum(gap["filled_samples"] for gap in gaps),
                        "trajectory_extent_m": np.ptp(output_positions, axis=0).tolist()})
    except ValueError as exc:
        quality["rejection_reason"] = str(exc)
        _save_json(quality_path, quality)
        raise
    result = {"schema_version": 1,
              "source": {"video": _portable_source_path(video_path), "sha256": _sha256(video_path),
                         "kind": "personal_video", "calibration_sha256": _sha256(calibration_path),
                         "events_sha256": _sha256(events_path)},
              "coordinate_frame": "table_marker", "units": "metres",
              "samples": [{"t": float(t), "xyz": xyz.tolist()} for t, xyz in zip(output_times, output_positions)],
              "grasp_time_s": events["grasp_time_s"], "release_time_s": events["release_time_s"],
              "target_xyz": target.tolist(),
              "markers": {"dictionary": ARUCO_DICTIONARY, "table_id": 0, "object_id": 1,
                          "table_marker_size_m": table_marker_size_m, "object_marker_size_m": object_marker_size_m},
              "quality": quality}
    _save_json(quality_path, quality)
    _save_json(output_path, result)
    return result
