"""Candidate A Phase-2 calibration: tap grid -> 1-D u->bearing fit.

Reads per-camera tap CSVs (schema in ../calib_grid.py), fits the 1-D
`u -> azimuth-bearing` polynomial that position.py's solve() consumes, and
writes the artifacts:
  - bearing_map_cam{0,1}.npz   keys: coef (poly1d, high->low), degree, cam_id
  - screen_config.yaml         corners + baseline + screen dims (norm & cm)

Bearing = azimuth from a camera's known top-corner to the tapped screen point,
in the position.py frame (origin top-left cm, +x right, +y down):
    bearing = atan2(y_cm - cy, x_cm - cx),  (x_cm,y_cm) = (x_norm*W, y_norm*H)

Round-trip: load_config(dir) rebuilds the exact screen_config dict solve() eats.

Run:  python calibrate.py [--taps-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np

# Physical screen (PROJECT.md: 30.4 x 19.7 cm active area, baseline = width).
W_CM, H_CM = 30.4, 19.7
# Cameras at the two TOP corners. Parameter/default; the rig geometry is known,
# so we do NOT refine these from taps.
# ponytail: corner refinement skipped — it's a nonlinear joint fit the degree-6
# poly largely absorbs anyway; add only if a bumped rig shows a corner offset.
CORNER0_CM = (0.0, 0.0)
CORNER1_CM = (W_CM, 0.0)
MAX_DEG = 6            # PROJECT.md caps the 1-D fit here
DT_THRESHOLD_US = 8000


def read_taps(path: Path):
    """taps_camN.csv -> (u[], x_norm[], y_norm[]). A uses u only."""
    u, xn, yn = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            u.append(float(row["u"]))
            xn.append(float(row["x_norm"]))
            yn.append(float(row["y_norm"]))
    return np.array(u), np.array(xn), np.array(yn)


def _bearings(xn, yn, corner_cm):
    x_cm, y_cm = xn * W_CM, yn * H_CM
    return np.arctan2(y_cm - corner_cm[1], x_cm - corner_cm[0])


def fit_bearing_map(u, bearings, max_deg=MAX_DEG):
    """Pick poly degree by held-out residual (cap max_deg), refit on all taps.

    Returns (poly1d, chosen_degree, holdout_rms_rad).
    """
    idx = np.argsort(u)                       # deterministic even/odd split by u
    train, test = idx[0::2], idx[1::2]
    best_deg, best_rms = 1, math.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        for deg in range(1, min(max_deg, len(train) - 1) + 1):
            p = np.poly1d(np.polyfit(u[train], bearings[train], deg))
            rms = float(np.sqrt(np.mean((p(u[test]) - bearings[test]) ** 2)))
            if rms < best_rms - 1e-9:          # prefer simpler on ties
                best_deg, best_rms = deg, rms
        p_full = np.poly1d(np.polyfit(u, bearings, best_deg))
    return p_full, best_deg, best_rms


def calibrate(taps_dir: Path, out_dir: Path,
              corner0=CORNER0_CM, corner1=CORNER1_CM):
    """Fit both cameras, write artifacts. Returns per-cam (degree, holdout_rms)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for cam, corner in ((0, corner0), (1, corner1)):
        u, xn, yn = read_taps(taps_dir / f"taps_cam{cam}.csv")
        b = _bearings(xn, yn, corner)
        poly, deg, rms = fit_bearing_map(u, b)
        np.savez(out_dir / f"bearing_map_cam{cam}.npz",
                 coef=poly.coefficients, degree=deg, cam_id=cam)
        stats[cam] = (deg, rms)
    _write_config(out_dir / "screen_config.yaml", corner0, corner1)
    return stats


# ---------------------------------------------------------------------------
# screen_config.yaml — flat scalar map so the reader needs no yaml dependency.
# ---------------------------------------------------------------------------
def _config_dict(corner0, corner1):
    return {
        "width_cm": W_CM, "height_cm": H_CM,
        "width_norm": 1.0, "height_norm": 1.0,
        "baseline_cm": corner1[0] - corner0[0],
        "corner0_x_cm": corner0[0], "corner0_y_cm": corner0[1],
        "corner1_x_cm": corner1[0], "corner1_y_cm": corner1[1],
        "dt_threshold_us": DT_THRESHOLD_US,
    }


def _write_config(path: Path, corner0, corner1):
    d = _config_dict(corner0, corner1)
    try:
        import yaml
        text = yaml.safe_dump(d, sort_keys=False)
    except ImportError:
        # ponytail: flat key: value only — no nesting to serialize.
        text = "".join(f"{k}: {v}\n" for k, v in d.items())
    path.write_text(text)


def _read_config(path: Path):
    try:
        import yaml
        return yaml.safe_load(path.read_text())
    except ImportError:
        d = {}
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            d[k.strip()] = float(v.strip())
        return d


def load_config(art_dir: Path):
    """Rebuild the screen_config dict that position.solve() consumes."""
    art_dir = Path(art_dir)
    cfg = _read_config(art_dir / "screen_config.yaml")
    m0 = np.load(art_dir / "bearing_map_cam0.npz")
    m1 = np.load(art_dir / "bearing_map_cam1.npz")
    return {
        "width_cm": cfg["width_cm"], "height_cm": cfg["height_cm"],
        "corner0": (cfg["corner0_x_cm"], cfg["corner0_y_cm"]),
        "corner1": (cfg["corner1_x_cm"], cfg["corner1_y_cm"]),
        "bearing_map0": np.poly1d(m0["coef"]),
        "bearing_map1": np.poly1d(m1["coef"]),
        "dt_threshold_us": cfg["dt_threshold_us"],
    }


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taps-dir", type=Path, default=here.parent,
                    help="dir holding taps_cam{0,1}.csv (default: repo C3/)")
    ap.add_argument("--out-dir", type=Path, default=here,
                    help="where to write artifacts (default: this dir)")
    a = ap.parse_args()
    stats = calibrate(a.taps_dir, a.out_dir)
    for cam, (deg, rms) in stats.items():
        print(f"cam{cam}: degree {deg}, held-out RMS {math.degrees(rms):.4f} deg")
    print(f"wrote bearing_map_cam0/1.npz + screen_config.yaml -> {a.out_dir}")


if __name__ == "__main__":
    main()
