"""Phase 1 — tap-calibration capture (Pi, dual camera). Candidate A.

Run on the Pi with the Mac showing show_grid.py fullscreen:
    python3 capture_calib.py               # tap grid -> taps_cam{0,1}.csv

Operator taps the numbered dots ON THE MAC SCREEN in order. When both cameras
see a STABLE held point, the averaged (u,v) is logged against that dot's known
(x,y). A needs both bearings, so a tap only one camera sees is unsolvable and
correctly skipped. Pen-up between taps arms the next capture.
Press 'u' to undo the last dot, 'q' to quit (partial calibration is saved).
"""
import csv

import cv2
import numpy as np

from calib_grid import grid_points, TAP_COLS
from detect import detect
from live_dual import open_cams, grab   # reuse locked-exposure dual-cam setup

STABLE_K = 8       # frames of low-variance detection to accept a tap
STABLE_TOL = 3.0   # px std over the window to count as "held still"


def _capture_taps():
    grid = grid_points()
    cams = open_cams()
    files = {c: open(f"taps_cam{c}.csv", "w", newline="") for c in (0, 1)}
    wr = {c: csv.writer(files[c]) for c in (0, 1)}
    for c in (0, 1):
        wr[c].writerow(TAP_COLS)
    recorded = {c: [] for c in (0, 1)}  # rows, for undo

    idx = 0
    buf = {0: [], 1: []}
    armed = True  # capture only after a pen-up since the last dot
    try:
        while idx < len(grid):
            got = {}
            for c in (0, 1):
                f, ts = grab(cams[c])
                got[c] = (f, detect(f), ts)
            both = got[0][1] and got[1][1]

            if both:
                for c in (0, 1):
                    buf[c].append(got[c][1][:2])
                    buf[c] = buf[c][-STABLE_K:]
                stable = (len(buf[0]) >= STABLE_K and
                          all((np.std(buf[c], 0) < STABLE_TOL).all() for c in (0, 1)))
                if armed and stable:
                    x, y = grid[idx]
                    for c in (0, 1):
                        a = np.array(buf[c]); mu = a.mean(0); sd = a.std(0)
                        row = [idx, f"{x:.5f}", f"{y:.5f}", f"{mu[0]:.2f}",
                               f"{mu[1]:.2f}", f"{sd[0]:.2f}", f"{sd[1]:.2f}",
                               len(a), got[c][2]]
                        wr[c].writerow(row); recorded[c].append(row)
                    print(f"dot {idx + 1}/{len(grid)} @ ({x:.2f},{y:.2f}) captured")
                    idx += 1; armed = False; buf = {0: [], 1: []}
            else:
                buf = {0: [], 1: []}
                if not (got[0][1] or got[1][1]):
                    armed = True  # both pen-up -> ready for next tap

            _show(got, idx, len(grid), armed)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("u") and idx > 0:  # undo last dot
                idx -= 1; armed = True; buf = {0: [], 1: []}
                for c in (0, 1):
                    recorded[c].pop()
                print(f"undo -> redo dot {idx + 1}")
    finally:
        # rewrite files from `recorded` so an undo actually drops the row
        for c in (0, 1):
            files[c].seek(0); files[c].truncate()
            wr[c].writerow(TAP_COLS)
            wr[c].writerows(recorded[c])
            files[c].close()
        cv2.destroyAllWindows()
        for cam in cams:
            cam.stop()
        print(f"saved taps_cam0.csv / taps_cam1.csv ({idx} dots)")


def _show(got, idx, total, armed):
    vis = []
    for c in (0, 1):
        f, r, _ = got[c]
        im = f.copy()
        if r:
            cv2.drawMarker(im, (int(r[0]), int(r[1])), (0, 255, 255),
                           cv2.MARKER_CROSS, 40, 3)
        cv2.putText(im, f"CAM {c}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        vis.append(im)
    combined = cv2.hconcat(vis)
    msg = (f"TAP DOT {idx + 1}/{total}  " + ("READY" if armed else "lift pen"))
    cv2.putText(combined, msg, (10, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if armed else (0, 165, 255), 2)
    if combined.shape[1] > 1600:
        s = 1600 / combined.shape[1]
        combined = cv2.resize(combined, None, fx=s, fy=s)
    cv2.imshow("calib", combined)


if __name__ == "__main__":
    _capture_taps()
