"""Dual-camera live LED tracker for the Pi (needs a display).

Run with SYSTEM python3 (picamera2 + GUI OpenCV), NOT the headless test venv:
    sudo apt install -y python3-opencv python3-picamera2
    python3 live_dual.py

Opens ONE window showing both cameras side by side, each with its blue-LED
detection marked. Cameras run at a fixed identical frame rate with locked
exposure/AWB (auto would drift them apart); each frame's SensorTimestamp is read
and the per-pair offset dt is shown + logged. Logs every pair to dual_log.csv for
triangulation + jitter analysis. Press 'q' to quit.
"""
import csv
import sys
import time

import cv2

from detect import detect

# --- locked camera settings (from bench tuning) ---
EXPOSURE_TIME_US = 800
ANALOGUE_GAIN = 1.0
COLOUR_GAINS = (2.0, 2.0)
LENS_POSITION = 5.0
SIZE = (1280, 720)
FPS = 100.0
DT_MAX_US = 8000       # accept a pair only if |SensorTimestamp0 - 1| < this
SWAP_RB = False        # set True if the LED shows RED / detection fails (RGB<->BGR)
DISPLAY_W = 1600       # downscale the side-by-side view to fit the screen


def apply_led_controls(camera, camera_index):
    """Lock exposure/AWB/focus so the two cameras stay deterministic + in step.
    libcamera is Pi-only -> imported lazily so this module still imports on Mac.
    """
    from libcamera import controls
    ctrls = {
        "AeEnable": False,          # no auto-exposure re-metering (drifts sync)
        "AwbEnable": False,         # lock white balance -> stable blue hue
        "AnalogueGain": ANALOGUE_GAIN,
        "ColourGains": COLOUR_GAINS,
        "ExposureTime": EXPOSURE_TIME_US,
    }
    try:
        ctrls["AfMode"] = controls.AfModeEnum.Manual
        ctrls["LensPosition"] = LENS_POSITION
    except Exception:
        pass
    try:
        ctrls["HdrMode"] = controls.HdrModeEnum.Off  # single exposure, no merge
    except Exception:
        pass
    camera.set_controls(ctrls)


def open_cams():
    from picamera2 import Picamera2
    cams = []
    for i in range(2):
        p = Picamera2(i)
        cfg = p.create_video_configuration(
            main={"size": SIZE, "format": "RGB888"},
            controls={"FrameRate": FPS},
        )
        p.configure(cfg)
        p.start()
        apply_led_controls(p, i)
        cams.append(p)
    print(f"opened {len(cams)} cameras @ {SIZE} {FPS}fps")
    return cams


def grab(cam):
    """Return (bgr_frame, sensor_timestamp_ns) for one camera, same frame."""
    req = cam.capture_request()
    try:
        arr = req.make_array("main")
        ts = req.get_metadata().get("SensorTimestamp", 0)
    finally:
        req.release()
    if SWAP_RB:
        arr = arr[..., ::-1].copy()
    return arr, ts


def annotate(bgr, res, label):
    vis = bgr.copy()
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if res:
        u, v, c = res
        cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.circle(vis, (int(u), int(v)), 26, (0, 255, 255), 2)
        cv2.putText(vis, f"u={u:.1f} v={v:.1f}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(vis, "no LED", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def main():
    cams = open_cams()
    f = open("dual_log.csv", "w", newline="")
    log = csv.writer(f)
    log.writerow(["frame", "t", "dt_us", "accept",
                  "u0", "v0", "c0", "u1", "v1", "c1"])
    t0 = time.monotonic()
    frame = 0
    # ponytail: back-to-back capture IS the pairing for a viewer; upgrade to a
    # buffered nearest-timestamp match if dt proves too loose.
    try:
        while True:
            (f0, ts0), (f1, ts1) = grab(cams[0]), grab(cams[1])
            r0, r1 = detect(f0), detect(f1)
            dt_us = abs(ts0 - ts1) / 1000.0
            accept = dt_us < DT_MAX_US
            t = time.monotonic() - t0

            v0 = annotate(f0, r0, "CAM 0")
            v1 = annotate(f1, r1, "CAM 1")
            combined = cv2.hconcat([v0, v1])
            color = (0, 255, 0) if accept else (0, 0, 255)
            cv2.putText(combined, f"dt={dt_us:.0f}us {'OK' if accept else 'REJECT'}",
                        (10, combined.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if combined.shape[1] > DISPLAY_W:
                s = DISPLAY_W / combined.shape[1]
                combined = cv2.resize(combined, None, fx=s, fy=s)
            # frames are BGR internally (what detect + cv2 expect) -> imshow shows
            # true color. If the LED looks RED, flip SWAP_RB at the top.
            cv2.imshow("dual", combined)

            def cols(r):
                return [f"{r[0]:.2f}", f"{r[1]:.2f}", f"{r[2]:.3f}"] if r else ["", "", ""]
            log.writerow([frame, f"{t:.4f}", f"{dt_us:.1f}", int(accept),
                          *cols(r0), *cols(r1)])
            frame += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        f.close()
        cv2.destroyAllWindows()
        for c in cams:
            c.stop()
        print(f"saved dual_log.csv ({frame} pairs)")


if __name__ == "__main__":
    main()
