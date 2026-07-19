"""Candidate A — end-to-end: calibrate (if needed) -> live (x,y) stream.

    python3 run_a.py               # calibrate if no artifacts, then run live
    python3 run_a.py --recalibrate # force a fresh calibration first
    python3 run_a.py --selftest    # no hardware: verify the wiring

Live: prints screen (x,y) every frame BOTH cameras see the LED. Press 'p' to
toggle the camera preview window (off = faster, headless-ish), 'q' to quit.
Also logs to xy_log.csv. Camera 0 = top-LEFT corner, camera 1 = top-RIGHT.
"""
import csv
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "A_triangulation"))

from detect import detect                        # noqa: E402
import calibrate as acal                          # A_triangulation/calibrate.py  # noqa: E402
from position import Detection, solve             # A_triangulation/position.py   # noqa: E402

ART_DIR = os.path.join(HERE, "A_triangulation")
DISPLAY_W = 1600


def _artifacts_exist():
    need = ["bearing_map_cam0.npz", "bearing_map_cam1.npz", "screen_config.yaml"]
    return all(os.path.exists(os.path.join(ART_DIR, f)) for f in need)


def calibrate_phase():
    """Run the tap-grid capture, then fit A's bearing maps."""
    import capture_calib
    print("=== CALIBRATION ===  tap the dots on the Mac grid (show_grid.py)")
    capture_calib._capture_taps()                 # writes taps_cam{0,1}.csv to CWD
    from pathlib import Path
    stats = acal.calibrate(Path(HERE), Path(ART_DIR))
    for cam, (deg, rms) in stats.items():
        print(f"  cam{cam}: degree {deg}, held-out RMS {np.degrees(rms):.4f} deg")
    print("calibration written.\n")


def _to_detection(cam_id, res, ts_ns):
    if res is None:
        return None
    u, v, conf = res
    return Detection(cam_id, u, v, ts_ns // 1000, conf)


def _render(frames, dets, sp, preview):
    if not preview:
        card = np.zeros((80, 480, 3), np.uint8)
        cv2.putText(card, "preview OFF  (p=on, q=quit)", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("run_a", card)
        return
    vis = []
    for c in (0, 1):
        im = frames[c].copy()
        cv2.putText(im, f"CAM {c}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        if dets[c] is not None:
            cv2.drawMarker(im, (int(dets[c].u), int(dets[c].v)),
                           (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
        vis.append(im)
    combined = cv2.hconcat(vis)
    txt = (f"x={sp.x_norm:.4f} y={sp.y_norm:.4f}" if sp else "no (x,y)")
    cv2.putText(combined, txt, (10, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if sp else (0, 0, 255), 2)
    if combined.shape[1] > DISPLAY_W:
        s = DISPLAY_W / combined.shape[1]
        combined = cv2.resize(combined, None, fx=s, fy=s)
    cv2.imshow("run_a", combined)


def live_phase():
    from live_dual import open_cams, grab
    cfg = acal.load_config(ART_DIR)
    cams = open_cams()
    f = open("xy_log.csv", "w", newline="")
    log = csv.writer(f)
    log.writerow(["t", "x_norm", "y_norm", "pen_down"])
    t0 = time.monotonic()
    preview = True
    print("=== LIVE ===  p=toggle preview, q=quit")
    try:
        while True:
            frames, dets = {}, {}
            for c in (0, 1):
                fr, ts = grab(cams[c])
                frames[c] = fr
                dets[c] = _to_detection(c, detect(fr), ts)
            sp = solve(dets[0], dets[1], cfg)
            t = time.monotonic() - t0
            if sp is not None:
                print(f"x={sp.x_norm:.4f} y={sp.y_norm:.4f} pen_down={sp.pen_down}")
                log.writerow([f"{t:.4f}", f"{sp.x_norm:.5f}",
                              f"{sp.y_norm:.5f}", int(sp.pen_down)])
            _render(frames, dets, sp, preview)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("p"):
                preview = not preview
                cv2.destroyWindow("run_a")
    finally:
        f.close()
        cv2.destroyAllWindows()
        for c in cams:
            c.stop()
        print("saved xy_log.csv")


def selftest():
    """No hardware: build a synthetic cfg + fake detections, exercise the wiring."""
    import math
    from pathlib import Path
    import tempfile
    from calib_grid import grid_points
    W, H = acal.W_CM, acal.H_CM
    C0, C1 = np.array(acal.CORNER0_CM), np.array(acal.CORNER1_CM)

    def true_u(b, cam):
        return 640 + 900 * b + 120 * b * b + (30 if cam else 0)

    d, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    for cam, corner in ((0, C0), (1, C1)):
        with open(f"{d}/taps_cam{cam}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dot", "x_norm", "y_norm", "u", "v", "u_std", "v_std", "n", "t_us"])
            for i, (xn, yn) in enumerate(grid_points()):
                b = math.atan2(yn * H - corner[1], xn * W - corner[0])
                w.writerow([i, f"{xn:.5f}", f"{yn:.5f}", f"{true_u(b, cam):.2f}",
                            0, 0, 0, 8, i * 1000])
    acal.calibrate(Path(d), Path(out))
    cfg = acal.load_config(out)

    # a known point -> fake detections -> solve -> should recover it
    xn, yn = 0.42, 0.6
    us = [true_u(math.atan2(yn * H - c[1], xn * W - c[0]), k)
          for k, c in ((0, C0), (1, C1))]
    d0 = _to_detection(0, (us[0], 0.0, 1.0), 1_000_000)
    d1 = _to_detection(1, (us[1], 0.0, 1.0), 1_000_000)
    sp = solve(d0, d1, cfg)
    assert sp is not None and abs(sp.x_norm - xn) < 0.02 and abs(sp.y_norm - yn) < 0.02, sp
    # one camera missing -> A drops out
    assert solve(d0, None, cfg) is None
    print(f"selftest OK: recovered ({sp.x_norm:.3f},{sp.y_norm:.3f}) vs ({xn},{yn})")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--recalibrate" in sys.argv or not _artifacts_exist():
        if not _artifacts_exist():
            print("no calibration artifacts found -> calibrating first")
        calibrate_phase()
    live_phase()


if __name__ == "__main__":
    main()
