"""Known-geometry tests for coplanar reference fitting and duplicate object IDs."""
import math
import unittest

import numpy as np

from phone_to_panda.multimarker import fit_reference_board, select_object_detection, reference_detections
from phone_to_panda.video import marker_object_points


class MultiMarkerTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.cv2 = cv2
        self.matrix = np.array([[950, 0, 640], [0, 950, 360], [0, 0, 1]], float)
        self.distortion = np.zeros(5)
        self.rvec = np.array([math.pi - .25, .03, -.02])
        self.rotation = cv2.Rodrigues(self.rvec)[0]
        self.translation = np.array([-.20, -.15, .8])
        self.size = [1280, 720]
        self.detections = []
        self.specs = []
        self.centers = [[0, 0, 0], [.40, 0, 0], [.40, -.30, 0], [0, -.30, 0]]
        for key, center in enumerate(self.centers):
            points = marker_object_points(.05) + center
            corners = cv2.projectPoints(points, self.rvec, self.translation, self.matrix, self.distortion)[0].reshape(4, 2)
            self.detections.append({"id": key, "corners": corners.tolist()})
            normalized = corners.mean(0) / self.size
            self.specs.append({"id": key, "region": np.r_[normalized - .065, normalized + .065].tolist()})
        self.object_xyz = np.array([.17, -.13, .04])
        object_t = self.rotation @ self.object_xyz + self.translation
        corners = cv2.projectPoints(marker_object_points(.03), self.rvec, object_t, self.matrix, self.distortion)[0].reshape(4, 2)
        self.object_detection = {"id": 1, "corners": corners.tolist()}

    def fitted_board(self):
        return fit_reference_board([{"detections": self.detections + [self.object_detection]}] * 4,
                                   self.specs, self.size, .05, self.matrix, self.distortion)

    def test_reference_board_recovers_known_metric_frame_without_given_spacing(self):
        board = self.fitted_board()
        np.testing.assert_allclose(board["camera_translation"], self.translation, atol=1e-6)
        np.testing.assert_allclose(board["camera_rotation"], self.rotation, atol=1e-6)
        for key, center in enumerate(self.centers):
            np.testing.assert_allclose(np.mean(board["reference_board_corners_m"][str(key)], axis=0), center, atol=1e-6)
        self.assertLess(board["rms_reprojection_error_px"], 1e-5)

    def test_duplicate_id_on_object_does_not_replace_reference(self):
        selected = reference_detections(self.detections + [self.object_detection], self.specs, self.size)
        np.testing.assert_allclose(selected[1], self.detections[1]["corners"])
        selected_object = select_object_detection(self.detections + [self.object_detection], self.fitted_board())
        np.testing.assert_allclose(selected_object, self.object_detection["corners"])

    def test_missing_object_is_not_replaced_by_static_marker_with_same_id(self):
        self.assertIsNone(select_object_detection(self.detections, self.fitted_board()))

    def test_multiple_object_candidates_fail_instead_of_arbitrary_choice(self):
        other = {"id": 1, "corners": (np.array(self.object_detection["corners"]) + [75, 0]).tolist()}
        self.assertIsNone(select_object_detection(self.detections + [self.object_detection, other], self.fitted_board()))

    def test_non_coplanar_reference_rejected(self):
        raised = dict(self.detections[3])
        points = marker_object_points(.05) + np.array(self.centers[3]) + [0, 0, .30]
        raised["corners"] = self.cv2.projectPoints(points, self.rvec, self.translation, self.matrix, self.distortion)[0].reshape(4, 2).tolist()
        # Broad regions permit the displaced marker; the metric plane must catch it.
        specs = [{"id": key, "region": [0, 0, 1, 1]} for key in range(4)]
        with self.assertRaisesRegex(ValueError, "inconsistent with one plane"):
            fit_reference_board([{"detections": self.detections[:3] + [raised]}] * 4, specs, self.size,
                                .05, self.matrix, self.distortion, max_reprojection_error_px=3)


if __name__ == "__main__":
    unittest.main()
