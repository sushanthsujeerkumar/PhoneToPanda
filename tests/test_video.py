"""Meaningful geometry and data-quality checks; no applicant video is fabricated."""

import math
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phone_to_panda.video import (
    camera_point_in_table, estimate_marker_pose, interpolate_short_gaps,
    marker_object_points, validate_events, generate_calibration_assets, extract_video,
)


class VideoGeometryTests(unittest.TestCase):
    def test_known_coordinate_transform(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        translation = np.array([0.3, -0.2, 0.8])
        expected = np.array([0.12, -0.07, 0.04])
        observed_camera = rotation @ expected + translation
        np.testing.assert_allclose(camera_point_in_table(observed_camera, rotation, translation), expected)

    def test_marker_pose_recovers_projected_metric_point(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        matrix = np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1]], dtype=float)
        distortion = np.zeros(5)
        rvec = np.array([math.pi - 0.25, 0.1, 0.03])
        expected_translation = np.array([0.08, 0.04, 0.65])
        points, _ = cv2.projectPoints(marker_object_points(0.08), rvec, expected_translation, matrix, distortion)
        rotation, translation, error = estimate_marker_pose(points, 0.08, matrix, distortion)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-5)
        np.testing.assert_allclose(rotation, cv2.Rodrigues(rvec)[0], atol=1e-5)
        self.assertLess(error, 1e-4)

    def test_negative_marker_size_rejected(self):
        with self.assertRaises(ValueError):
            marker_object_points(-0.03)

    def test_internal_short_occlusion_interpolates_at_actual_times(self):
        times = [0, 0.08, 0.18, 0.3]
        positions = [[0, 0, 0], [np.nan, np.nan, np.nan], [0.18, 0.36, 0.09], [0.3, 0.6, 0.15]]
        result, gaps = interpolate_short_gaps(times, positions, 0.2)
        np.testing.assert_allclose(result[1], [0.08, 0.16, 0.04])
        self.assertEqual(gaps[0]["filled_samples"], 1)

    def test_long_occlusion_rejected(self):
        with self.assertRaisesRegex(ValueError, "occlusion"):
            interpolate_short_gaps([0, 0.5, 1], [[0, 0, 0], [np.nan] * 3, [1, 1, 1]], 0.2)

    def test_endpoint_occlusion_never_extrapolated(self):
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            interpolate_short_gaps([0, 0.1], [[np.nan] * 3, [1, 1, 1]])

    def test_repeated_timestamps_rejected(self):
        with self.assertRaisesRegex(ValueError, "increasing"):
            interpolate_short_gaps([0, 0], [[0, 0, 0], [1, 1, 1]])


class EventValidationTests(unittest.TestCase):
    def test_null_contact_times_rejected(self):
        with self.assertRaisesRegex(ValueError, "grasp_time_s"):
            validate_events({"grasp_time_s": None, "release_time_s": None}, 5)

    def test_reversed_contact_times_rejected(self):
        with self.assertRaisesRegex(ValueError, "start < grasp"):
            validate_events({"grasp_time_s": 3, "release_time_s": 2}, 5)

    def test_events_keep_absolute_video_times(self):
        result = validate_events({"start_time_s": 1, "grasp_time_s": 2, "release_time_s": 4, "end_time_s": None}, 6)
        self.assertEqual(result["grasp_time_s"], 2)
        self.assertEqual(result["end_time_s"], 4)

    def test_out_of_video_release_rejected(self):
        with self.assertRaises(ValueError):
            validate_events({"grasp_time_s": 2, "release_time_s": 7}, 6)


class VideoExtractionTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        self.cv2 = cv2

    def test_printable_marker_assets_detect_correct_ids(self):
        cv2 = self.cv2
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary)
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_calibration_assets(directory)
            self.assertEqual(manifest["table_marker_size_m"], 0.05)
            self.assertEqual(manifest["object_marker_size_m"], 0.03)
            for marker_id, key in [(i, f"table_marker_{i}") for i in range(4)] + [(1, "object_marker_1")]:
                image = cv2.imread(manifest["files"][key])
                detected, ids, _ = detector.detectMarkers(image)
                self.assertEqual(ids.ravel().tolist(), [marker_id])
                points = detected[0].reshape(4, 2)
                # The printed image's canonical corners really are TL,TR,BR,BL:
                # pixel y runs down, while the documented table-frame y runs up.
                self.assertLess(points[0, 0], points[1, 0])
                self.assertLess(points[0, 1], points[3, 1])
                self.assertGreater(points[2, 0], points[3, 0])
                self.assertIn(b"pHYs", Path(manifest["files"][key]).read_bytes())

    def test_extraction_recovers_known_motion_and_writes_diagnostics(self):
        # This temporary rendered fixture tests geometry and codec handling only.
        # It is not retained as personal data, a demonstration, or a success result.
        cv2 = self.cv2
        size = (1280, 720)
        matrix = np.array([[950, 0, 640], [0, 950, 360], [0, 0, 1]], dtype=float)
        distortion = np.zeros(5)
        table_rvec = np.array([math.pi - 0.4, 0.03, 0.0])
        table_rotation = cv2.Rodrigues(table_rvec)[0]
        table_translation = np.array([-0.12, 0.02, 0.6])
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        corners = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            video = directory / "synthetic_geometry_fixture.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, size)
            self.assertTrue(writer.isOpened())
            for index in range(41):
                t = index / 20
                xyz = np.array([0.14 + t * 0.025, -0.06, 0.04 + 0.03 * math.sin(t)])
                frame = np.full((size[1], size[0]), 255, dtype=np.uint8)
                for marker_id, marker_size, translation in [
                    (0, 0.08, table_translation),
                    (1, 0.03, table_rotation @ xyz + table_translation),
                ]:
                    projected, _ = cv2.projectPoints(marker_object_points(marker_size), table_rvec,
                                                      translation, matrix, distortion)
                    transform = cv2.getPerspectiveTransform(corners, projected.reshape(4, 2).astype(np.float32))
                    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)
                    warped = cv2.warpPerspective(marker, transform, size, borderValue=255)
                    frame = np.minimum(frame, warped)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            writer.release()
            calibration = directory / "calibration.json"
            calibration.write_text(json.dumps({"accepted": True, "camera_matrix": matrix.tolist(),
                                                "distortion_coefficients": distortion.tolist(), "image_size": list(size)}))
            events = directory / "events.json"
            events.write_text(json.dumps({"start_time_s": 0.1, "grasp_time_s": 0.4,
                                         "release_time_s": 1.6, "end_time_s": 1.8}))
            result = extract_video(video, calibration, events, directory / "trajectory.json", table_marker_size_m=0.08)
            self.assertTrue(result["quality"]["accepted"])
            self.assertGreater(result["quality"]["measured_fraction"], 0.95)
            self.assertTrue((directory / "trajectory.preview.mp4").is_file())
            self.assertTrue((directory / "trajectory.quality.json").is_file())
            expected_target = [0.18, -0.06, 0.04 + 0.03 * math.sin(1.6)]
            np.testing.assert_allclose(result["target_xyz"], expected_target, atol=0.012)
            with self.assertRaisesRegex(FileExistsError, "new filename"):
                extract_video(video, calibration, events, directory / "trajectory.json", table_marker_size_m=0.08)


if __name__ == "__main__":
    unittest.main()
