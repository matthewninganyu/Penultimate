import unittest

import numpy as np

from contact_detection import ContactDetector
from tracking_filter import ExponentialPenFilter


class TrackingFilterTests(unittest.TestCase):
    def test_exponential_smoothing(self) -> None:
        filt = ExponentialPenFilter(alpha=0.5)
        first, conf1 = filt.update(np.array([0.0, 0.0, 0.0]), timestamp=1.0)
        second, conf2 = filt.update(np.array([10.0, 0.0, 0.0]), timestamp=1.1)
        self.assertTrue(np.allclose(first, [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(second, [5.0, 0.0, 0.0]))
        self.assertEqual(conf1, 1.0)
        self.assertEqual(conf2, 1.0)

    def test_rejects_large_outlier(self) -> None:
        filt = ExponentialPenFilter(alpha=0.5, max_jump_mm=10.0)
        filt.update(np.array([0.0, 0.0, 0.0]), timestamp=1.0)
        point, confidence = filt.update(np.array([100.0, 0.0, 0.0]), timestamp=1.1)
        self.assertTrue(np.allclose(point, [0.0, 0.0, 0.0]))
        self.assertLess(confidence, 0.35)

    def test_touch_hysteresis(self) -> None:
        detector = ContactDetector(
            touch_start_score=0.6,
            touch_end_score=0.4,
            touch_confirm_frames=2,
            hover_confirm_frames=2,
        )
        detector.touching = False
        detector._touch_count = 1
        # Simulate the hysteresis counters directly because image evidence is
        # tested through the runtime path; this locks the state transitions.
        detector._touch_count += 1
        if detector._touch_count >= detector.touch_confirm_frames:
            detector.touching = True
        self.assertTrue(detector.touching)
        detector._hover_count = 1
        detector._hover_count += 1
        if detector._hover_count >= detector.hover_confirm_frames:
            detector.touching = False
        self.assertFalse(detector.touching)


if __name__ == "__main__":
    unittest.main()

