"""Diagnose why A's calibrate RMS is high. Run on the Pi (or wherever the CSVs are):
    python3 diag_taps.py            # reads ./taps_cam{0,1}.csv

Tells you whether the residual comes from a few bad taps (mislabel / wrong blob)
or is spread across all of them (noise / geometry). Prints the worst offenders
so you can eyeball those dots.
"""
import csv
import math
import sys

import numpy as np

W_CM, H_CM = 30.4, 19.7
CORNERS = {0: (0.0, 0.0), 1: (W_CM, 0.0)}


def load(cam):
    rows = list(csv.DictReader(open(f"taps_cam{cam}.csv")))
    dot = np.array([int(r["dot"]) for r in rows])
    xn = np.array([float(r["x_norm"]) for r in rows])
    yn = np.array([float(r["y_norm"]) for r in rows])
    u = np.array([float(r["u"]) for r in rows])
    ustd = np.array([float(r["u_std"]) for r in rows])
    return dot, xn, yn, u, ustd


def bearings(xn, yn, corner):
    return np.arctan2(yn * H_CM - corner[1], xn * W_CM - corner[0])


for cam in (0, 1):
    try:
        dot, xn, yn, u, ustd = load(cam)
    except FileNotFoundError:
        sys.exit(f"taps_cam{cam}.csv not found — run from the dir holding the CSVs")
    b = bearings(xn, yn, CORNERS[cam])

    print(f"\n===== cam{cam}  ({len(u)} taps) =====")
    print(f"centroid jitter u_std: mean {ustd.mean():.2f}px  max {ustd.max():.2f}px "
          f"(capture gate was 3px)")

    # Monotonicity: sort by bearing, count how often u goes the 'wrong' way.
    order = np.argsort(b)
    du = np.diff(u[order])
    inversions = int(np.sum(du * np.sign(np.median(du)) < 0))
    print(f"u-vs-bearing monotonic? {len(u)-1-inversions}/{len(u)-1} steps consistent "
          f"({inversions} inversions — >a few means scrambled pairs)")

    # Full-degree fit, per-tap residual in degrees.
    p = np.poly1d(np.polyfit(u, b, min(6, len(u) - 2)))
    resid_deg = np.degrees(np.abs(p(u) - b))
    print(f"deg-6 fit residual: mean {resid_deg.mean():.3f}°  max {resid_deg.max():.3f}°")

    worst = np.argsort(resid_deg)[::-1][:5]
    print("worst 5 taps (dot#  x_norm  u px  u_std  resid°):")
    for i in worst:
        print(f"   dot {dot[i]:2d}   x={xn[i]:.3f}  u={u[i]:7.1f}  "
              f"u_std={ustd[i]:.2f}  resid={resid_deg[i]:.3f}°")

    # Verdict heuristic.
    spread = resid_deg.mean()
    concentrated = resid_deg[worst].mean() > 4 * np.median(resid_deg)
    if ustd.mean() > 3:
        print("  -> VERDICT: centroid jitter high — detector unstable (wrong/dim blob?).")
    elif concentrated:
        print("  -> VERDICT: a FEW taps dominate — mislabeled dots or wrong-blob picks. "
              "Recapture those, or fix tap order.")
    elif inversions > len(u) // 5:
        print("  -> VERDICT: pairs scrambled across the board — tap-order desync or "
              "u not monotonic (camera rotated?).")
    else:
        print("  -> VERDICT: residual spread evenly & low jitter — geometry model off "
              "(corner positions? screen dims?), not the taps.")
