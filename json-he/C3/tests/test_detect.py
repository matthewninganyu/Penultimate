"""Black-box test for the brightest-blue detector on the 5 real dark frames.

Run: python tests/test_detect.py   (or: pytest tests/test_detect.py)

Coords were verified by overlaying detect() output on each frame (tests/debug/
det_*.png) and confirming the marker sits on the blue LED. u tolerance is tighter
than v — u accuracy matters more for triangulation.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detect import detect  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
TOL_U = 40   # px — u matters more
TOL_V = 70   # px — v looser

# (u, v) of the LED, human-verified against tests/debug/det_*.png
EXPECTED = {
    "tl.png": (756, 720),
    "bl.png": (1175, 637),
    "mid.png": (581, 675),
    "tr.png": (208, 722),
    "br.png": (560, 683),  # dimmest frame (conf ~0.19) — still locks
}


def test_led_on_real_frames():
    for name, (eu, ev) in EXPECTED.items():
        im = cv2.imread(os.path.join(FIX, name))
        assert im is not None, f"missing fixture {name}"
        res = detect(im)
        assert res is not None, f"{name}: detected nothing"
        u, v, conf = res
        assert abs(u - eu) <= TOL_U, f"{name}: u={u:.0f} exp {eu} (±{TOL_U})"
        assert abs(v - ev) <= TOL_V, f"{name}: v={v:.0f} exp {ev} (±{TOL_V})"
        assert conf > 0.0


def test_pen_up_returns_none():
    assert detect(np.zeros((966, 1280, 3), np.uint8)) is None


def test_white_glare_rejected():
    # Bright neutral blob (glare) is not blue -> None.
    im = np.zeros((966, 1280, 3), np.uint8)
    cv2.circle(im, (640, 700), 60, (255, 255, 255), -1)
    assert detect(im) is None


def test_synthetic_blue_blob_centroid():
    # A blue blob is found near its center.
    im = np.zeros((966, 1280, 3), np.uint8)
    cv2.circle(im, (700, 600), 30, (255, 40, 0), -1)  # BGR: bright blue
    res = detect(im)
    assert res is not None
    u, v, _ = res
    assert abs(u - 700) <= 15 and abs(v - 600) <= 15


if __name__ == "__main__":
    test_led_on_real_frames()
    test_pen_up_returns_none()
    test_white_glare_rejected()
    test_synthetic_blue_blob_centroid()
    print("test_detect: ALL PASS")
