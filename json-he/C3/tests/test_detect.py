"""Black-box test for the shared blue-LED detector on real sample frames.

Run: python C3/tests/test_detect.py   (or: pytest C3/tests/test_detect.py)

Ground-truth pixel coords were verified by overlaying the detector output on
each frame and confirming the marker sits on the pen-tip contact (see
tests/debug/). They prove the detector rejects the blue pen barrel (above),
the reflection streak (below), and white overhead glare.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detect import detect  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
TOL = 45  # px; asserts the detector lands on the contact blob, not a distractor

# (u, v) of the LED contact, human-verified against the debug overlays.
EXPECTED = {
    "mid.png": (1005, 894),
    "br.png": (818, 941),
    "bl.png": (1729, 799),
    "TR.png": (284, 1017),
}


def test_contact_on_real_frames():
    for name, (eu, ev) in EXPECTED.items():
        im = cv2.imread(os.path.join(FIX, name))
        assert im is not None, f"missing fixture {name}"
        res = detect(im)
        assert res is not None, f"{name}: detected nothing"
        u, v, conf = res
        assert abs(u - eu) <= TOL and abs(v - ev) <= TOL, (
            f"{name}: got ({u:.0f},{v:.0f}) exp ({eu},{ev}) — wrong blob "
            f"(barrel/reflection/glare?)"
        )
        assert conf > 0.5, f"{name}: low confidence {conf:.2f}"


def test_pen_up_returns_none():
    # No LED at all -> pen up.
    assert detect(np.zeros((1422, 1852, 3), np.uint8)) is None


def test_white_glare_rejected():
    # Bright neutral blob (overhead-light glare) is not blue -> None.
    im = np.zeros((1422, 1852, 3), np.uint8)
    cv2.circle(im, (900, 1000), 80, (255, 255, 255), -1)
    assert detect(im) is None


def test_dim_blue_below_floor():
    # Faint blue (background UI, not the LED) stays under the present-floor.
    im = np.zeros((1422, 1852, 3), np.uint8)
    cv2.circle(im, (900, 1000), 60, (120, 0, 0), -1)  # BGR: dim blue
    assert detect(im) is None


if __name__ == "__main__":
    test_contact_on_real_frames()
    test_pen_up_returns_none()
    test_white_glare_rejected()
    test_dim_blue_below_floor()
    print("test_detect: ALL PASS")
