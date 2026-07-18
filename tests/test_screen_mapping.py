import unittest

import numpy as np

from models import ScreenCalibration
from screen_mapping import point_to_screen_position


def calibration() -> ScreenCalibration:
    eye = np.eye(3)
    return ScreenCalibration(
        K0=eye,
        D0=np.zeros(5),
        R0=eye,
        t0=np.array([[0.0], [0.0], [500.0]]),
        P0=np.hstack((eye, np.array([[0.0], [0.0], [500.0]]))),
        K1=eye,
        D1=np.zeros(5),
        R1=eye,
        t1=np.array([[100.0], [0.0], [500.0]]),
        P1=np.hstack((eye, np.array([[100.0], [0.0], [500.0]]))),
        screen_width_mm=344.0,
        screen_height_mm=194.0,
        screen_width_px=1920,
        screen_height_px=1080,
        image_width=640,
        image_height=480,
        reprojection_error_0=0.2,
        reprojection_error_1=0.2,
        timestamp=1.0,
    )


class ScreenMappingTests(unittest.TestCase):
    def test_maps_mm_to_normalized_and_pixels(self) -> None:
        position = point_to_screen_position(np.array([172.0, 97.0, 2.8]), calibration())
        self.assertIsNotNone(position)
        assert position is not None
        self.assertAlmostEqual(position.normalized_x, 0.5)
        self.assertAlmostEqual(position.normalized_y, 0.5)
        self.assertEqual(position.pixel_x, 960)
        self.assertEqual(position.pixel_y, 540)
        self.assertAlmostEqual(position.distance_mm, 2.8)

    def test_rejects_far_outside_screen(self) -> None:
        self.assertIsNone(point_to_screen_position(np.array([-50.0, 20.0, 1.0]), calibration(), margin_mm=10.0))

    def test_clamps_slightly_outside_screen(self) -> None:
        position = point_to_screen_position(np.array([-3.0, 196.0, 1.0]), calibration(), margin_mm=10.0)
        self.assertIsNotNone(position)
        assert position is not None
        self.assertEqual(position.normalized_x, 0.0)
        self.assertEqual(position.normalized_y, 1.0)


if __name__ == "__main__":
    unittest.main()

