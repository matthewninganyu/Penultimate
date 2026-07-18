import unittest

import cv2
import numpy as np

from models import LedCandidate, ScreenCalibration
from stereo_matching import choose_best_stereo_match
from triangulation import reprojection_error


def candidate(x: float, y: float, area: float = 50.0) -> LedCandidate:
    contour = np.array([[[int(x), int(y)]], [[int(x + 2), int(y)]], [[int(x + 2), int(y + 2)]], [[int(x), int(y + 2)]]])
    return LedCandidate(
        x=x,
        y=y,
        contour_x=x,
        contour_y=y,
        peak_x=x,
        peak_y=y,
        area=area,
        radius=4.0,
        width=4.0,
        height=4.0,
        circularity=1.0,
        mean_brightness=240.0,
        peak_brightness=255.0,
        contour=contour,
    )


def synthetic_calibration() -> ScreenCalibration:
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5)
    R0 = np.eye(3)
    t0 = np.array([[0.0], [0.0], [500.0]])
    R1 = np.eye(3)
    t1 = np.array([[120.0], [0.0], [500.0]])
    return ScreenCalibration(
        K0=K,
        D0=D,
        R0=R0,
        t0=t0,
        P0=K @ np.hstack((R0, t0)),
        K1=K,
        D1=D,
        R1=R1,
        t1=t1,
        P1=K @ np.hstack((R1, t1)),
        screen_width_mm=344.0,
        screen_height_mm=194.0,
        screen_width_px=1920,
        screen_height_px=1080,
        image_width=640,
        image_height=480,
        reprojection_error_0=0.1,
        reprojection_error_1=0.1,
        timestamp=1.0,
    )


class StereoMatchingTests(unittest.TestCase):
    def test_reprojection_error_calculation(self) -> None:
        cal = synthetic_calibration()
        point = np.array([100.0, 50.0, 5.0])
        rvec0, _ = cv2.Rodrigues(cal.R0)
        rvec1, _ = cv2.Rodrigues(cal.R1)
        p0, _ = cv2.projectPoints(point.reshape(1, 3), rvec0, cal.t0, cal.K0, cal.D0)
        p1, _ = cv2.projectPoints(point.reshape(1, 3), rvec1, cal.t1, cal.K1, cal.D1)
        err = reprojection_error(point, candidate(*p0.reshape(2)), candidate(*p1.reshape(2)), cal)
        self.assertLess(err, 1e-6)

    def test_candidate_pair_scoring_selects_geometric_pair(self) -> None:
        cal = synthetic_calibration()
        point = np.array([100.0, 50.0, 5.0])
        rvec0, _ = cv2.Rodrigues(cal.R0)
        rvec1, _ = cv2.Rodrigues(cal.R1)
        p0, _ = cv2.projectPoints(point.reshape(1, 3), rvec0, cal.t0, cal.K0, cal.D0)
        p1, _ = cv2.projectPoints(point.reshape(1, 3), rvec1, cal.t1, cal.K1, cal.D1)
        good0 = candidate(*p0.reshape(2))
        good1 = candidate(*p1.reshape(2))
        bad0 = candidate(20.0, 20.0)
        bad1 = candidate(600.0, 450.0)
        match = choose_best_stereo_match([bad0, good0], [bad1, good1], cal)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertIs(match.camera_0_candidate, good0)
        self.assertIs(match.camera_1_candidate, good1)


if __name__ == "__main__":
    unittest.main()

