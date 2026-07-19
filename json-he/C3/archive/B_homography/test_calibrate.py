"""Integration test for B's Phase-2 calibrate.py.

Covers the B-specific novel path: synthetic taps through known K+distortion ->
fit_homography -> load_calib -> position.solve() recovers held-out points.

Intrinsics recovery (calibrate_intrinsics) is stock cv2.calibrateCamera +
cv2.fisheye.calibrate; it is validated on-rig with REAL checkerboard images (the
R1 straight-line gate in PROJECT.md), not synthetically here — a synthetic board
clean enough for findChessboardCorners tests OpenCV, not our code.
Run: python3 test_calibrate.py
"""
import csv
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import calibrate as C          # noqa: E402
from position import solve     # noqa: E402
from calib_grid import grid_points  # noqa: E402

_S = 1280 / 4608.0
K_TRUE = np.array([[1995.7 * _S, 0, 2340.0 * _S],
                   [0, 1990.1 * _S, 1289.1 * _S], [0, 0, 1]])
DIST_TRUE = np.array([-0.0574, 0.1264, -0.00287, 0.00384, -0.0808])
RES = (1280, 720)
RVEC = np.array([1.35, 0.02, 0.02])   # grazing top-corner view
TVEC = np.array([0.05, 0.02, 0.30])


def _screen_to_px(xn, yn):
    P = np.array([[xn * 0.304, yn * 0.197, 0.0]], np.float64)
    px, _ = cv2.projectPoints(P, RVEC, TVEC, K_TRUE, DIST_TRUE)
    return px.reshape(2)


def test_no_images_returns_none():
    # Graceful skip when no checkerboard imgs yet (real pre-rig state).
    assert C.calibrate_intrinsics(tempfile.mkdtemp()) is None


def test_homography_roundtrip():
    rng = np.random.default_rng(0)
    taps, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    with open(f"{taps}/taps_cam0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dot", "x_norm", "y_norm", "u", "v", "u_std", "v_std", "n", "t_us"])
        for i, (xn, yn) in enumerate(grid_points()):
            u, v = _screen_to_px(xn, yn)
            w.writerow([i, f"{xn:.5f}", f"{yn:.5f}", f"{u + rng.normal(0, 0.3):.2f}",
                        f"{v + rng.normal(0, 0.3):.2f}", 0, 0, 8, i * 1000])
    np.savez(f"{out}/intrinsics_cam0.npz", K=K_TRUE, dist=DIST_TRUE, model="pinhole")
    H = C.fit_homography(f"{taps}/taps_cam0.csv", K_TRUE, DIST_TRUE, "pinhole")
    np.save(f"{out}/homography_cam0.npy", H)
    calib = C.load_calib(out, cams=(0,))
    assert set(calib["cams"][0]) == {"K", "dist", "H", "model"}

    errs = []
    for _ in range(300):
        xn, yn = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        u, v = _screen_to_px(xn, yn)
        sp = solve({"cam_id": 0, "u": float(u), "v": float(v),
                    "timestamp_us": 1000, "confidence": 1.0}, None, calib)
        errs.append(np.hypot((sp["x_norm"] - xn) * 304, (sp["y_norm"] - yn) * 197))
    errs = np.array(errs)
    assert errs.mean() < 3.0, f"held-out mean {errs.mean():.3f}mm too high"
    print(f"B homography round-trip: mean={errs.mean():.3f}mm max={errs.max():.3f}mm")


if __name__ == "__main__":
    test_no_images_returns_none()
    test_homography_roundtrip()
    print("test_calibrate (B): PASS")
