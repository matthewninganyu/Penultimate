"""Calibration dot grid (normalized [0,1]) + tap CSV schema. Candidate A.

Both show_grid.py (Mac display) and capture_calib.py (Pi logger) import this so
the two machines agree on dot positions AND order. Order is boustrophedon (snake)
so the operator taps with minimal travel and pen-up gaps mark tap boundaries.
"""
GRID_COLS = 3
GRID_ROWS = 3
MARGIN = 0.04  # keep dots off the extreme bezel but still sample the edges

# taps_cam{N}.csv columns (one row per captured dot per camera)
TAP_COLS = ["dot", "x_norm", "y_norm", "u", "v", "u_std", "v_std", "n", "t_us"]


def grid_points():
    """Ordered list of (x_norm, y_norm) dots, snake order."""
    import numpy as np
    xs = np.linspace(MARGIN, 1 - MARGIN, GRID_COLS)
    ys = np.linspace(MARGIN, 1 - MARGIN, GRID_ROWS)
    pts = []
    for j, y in enumerate(ys):
        cols = range(GRID_COLS) if j % 2 == 0 else range(GRID_COLS - 1, -1, -1)
        for i in cols:
            pts.append((float(xs[i]), float(y)))
    return pts


if __name__ == "__main__":
    g = grid_points()
    print(f"{len(g)} dots ({GRID_COLS}x{GRID_ROWS}), snake order")
    print("first 3:", g[:3])
