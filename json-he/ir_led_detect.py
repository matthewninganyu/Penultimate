#!/usr/bin/env python3
"""
IR LED detector for Raspberry Pi 5 + Camera Module 3 (IMX708).

Detects the bright point source of a ~730nm LED, encircles it, and reports a
per-blob "IR score" = red-dominance of the spot. An RGB Bayer sensor CANNOT
measure wavelength; it only reports channel response. A 730nm source on this
sensor reads red-dominant with weak G/B -> that ratio is the signature here.

Standard Module 3 (IR-cut filter) attenuates 730nm hard -> expect a faint or
absent dot. Module 3 NoIR passes it easily -> clear dot.

Controls:  q quit | e cycle exposure | m toggle mask view | s save frame
Run:  python3 ir_led_detect.py
Deps: sudo apt install -y python3-picamera2 python3-opencv
"""

import time
import numpy as np
import cv2
from picamera2 import Picamera2

# --- tunables -----------------------------------------------------------------
EXPOSURES_US = [200, 1000, 5000, 20000, 60000]  # cycle with 'e'; low = see only bright IR dot
GAIN = 4.0                 # analogue gain; raise if LED too dim, lower if noisy
BRIGHT_PCTILE = 99.7       # a pixel is "hot" if brighter than this percentile
MIN_BLOB_AREA = 6          # px; reject single-pixel noise
IR_SCORE_MIN = 1.15        # R/max(G,B) above this = red-dominant -> flag as IR-like
# ponytail: percentile threshold + red-ratio is the whole detector. If ambient
# light or reflections cause false hits, tighten BRIGHT_PCTILE / IR_SCORE_MIN.
# ------------------------------------------------------------------------------


def start_camera():
    cam = Picamera2()
    cfg = cam.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
    cam.configure(cfg)
    cam.start()
    time.sleep(0.5)
    return cam


def set_exposure(cam, exp_us):
    # manual exposure so the dim dot isn't blown out or hunted by auto-exposure
    cam.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": int(exp_us),
        "AnalogueGain": GAIN,
    })


def find_hot_blobs(bgr):
    """Return list of (cx, cy, radius, R, G, B, ir_score) for bright spots."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = np.percentile(gray, BRIGHT_PCTILE)
    thr = max(thr, 40)  # floor: ignore a dark frame's own noise ceiling
    _, mask = cv2.threshold(gray, int(thr), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < MIN_BLOB_AREA:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        blob = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        b, g, rr = (cv2.mean(bgr, mask=blob)[i] for i in range(3))
        ir_score = rr / max(g, b, 1.0)   # red dominance
        out.append((x, y, r, rr, g, b, ir_score))
    return mask, out


def main():
    cam = start_camera()
    exp_idx = 1
    set_exposure(cam, EXPOSURES_US[exp_idx])
    show_mask = False

    while True:
        rgb = cam.capture_array()                 # RGB888
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask, blobs = find_hot_blobs(bgr)

        view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else bgr.copy()
        ir_hits = 0
        for (x, y, r, R, G, B, score) in blobs:
            is_ir = score >= IR_SCORE_MIN
            ir_hits += is_ir
            color = (0, 0, 255) if is_ir else (0, 255, 255)  # red = IR-like, yellow = generic bright
            cv2.circle(view, (int(x), int(y)), int(r) + 8, color, 2)
            cv2.putText(view, f"IR={score:.2f} R{int(R)} G{int(G)} B{int(B)}",
                        (int(x) + 12, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        banner = f"exp={EXPOSURES_US[exp_idx]}us gain={GAIN}  IR-like blobs: {ir_hits}"
        cv2.putText(view, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if ir_hits:
            cv2.putText(view, "730nm DETECTED", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("IR LED detect", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("e"):
            exp_idx = (exp_idx + 1) % len(EXPOSURES_US)
            set_exposure(cam, EXPOSURES_US[exp_idx])
        elif k == ord("m"):
            show_mask = not show_mask
        elif k == ord("s"):
            cv2.imwrite(f"ir_capture_{int(time.time())}.png", view)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
