"""Integration test for A's Phase-2 calibrate.py: synthetic taps -> fit -> solve.

Confirms calibrate.py writes artifacts that load_config + position.solve() consume,
and that held-out points are recovered. Run: python3 test_calibrate.py
"""
import csv
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import calibrate as C          # noqa: E402
from position import Detection, solve  # noqa: E402
from calib_grid import grid_points     # noqa: E402

W, H = C.W_CM, C.H_CM
C0, C1 = np.array(C.CORNER0_CM), np.array(C.CORNER1_CM)


def _true_u(b, cam):
    # arbitrary monotonic lens map bearing->pixel (invertible), offset per cam
    return 640 + 900 * b + 120 * b * b + (30 if cam else 0)


def _write_synth_taps(d, rng, noise=0.5):
    for cam, corner in ((0, C0), (1, C1)):
        with open(f"{d}/taps_cam{cam}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dot", "x_norm", "y_norm", "u", "v", "u_std", "v_std", "n", "t_us"])
            for i, (xn, yn) in enumerate(grid_points()):
                b = math.atan2(yn * H - corner[1], xn * W - corner[0])
                u = _true_u(b, cam) + rng.normal(0, noise)
                w.writerow([i, f"{xn:.5f}", f"{yn:.5f}", f"{u:.2f}", 0, 0, 0, 8, i * 1000])


def test_fit_load_solve_roundtrip():
    rng = np.random.default_rng(0)
    taps = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    _write_synth_taps(taps, rng)

    C.calibrate(Path(taps), Path(out))
    cfg = C.load_config(out)
    assert {"bearing_map0", "bearing_map1", "corner0", "corner1",
            "width_cm", "height_cm", "dt_threshold_us"} <= set(cfg)

    errs = []
    for _ in range(300):
        xn, yn = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        us = [_true_u(math.atan2(yn * H - c[1], xn * W - c[0]), k)
              for k, c in ((0, C0), (1, C1))]
        sp = solve(Detection(0, us[0], 0, 1000, 1.0),
                   Detection(1, us[1], 0, 1000, 1.0), cfg)
        errs.append(math.hypot((sp.x_norm - xn) * W * 10, (sp.y_norm - yn) * H * 10))
    errs = np.array(errs)
    assert errs.mean() < 1.0, f"held-out mean {errs.mean():.3f}mm too high"
    print(f"A calibrate round-trip: mean={errs.mean():.3f}mm max={errs.max():.3f}mm")


if __name__ == "__main__":
    test_fit_load_solve_roundtrip()
    print("test_calibrate (A): PASS")
