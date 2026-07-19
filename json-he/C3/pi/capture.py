"""Capture real 720p frames on the Raspberry Pi for detector tuning + floor check.

Run ON the Pi, per camera (top-left = 0, top-right = 1):
    python3 capture.py --cam 0 --out frames_cam0

Press ENTER to save a frame, type 'q'+ENTER to quit. Capture, in order:
    1. ONE pen-UP frame  (LED OFF / pen lifted)  -> validates _PRESENT_FLOOR
    2. Several pen-DOWN frames at different screen spots (corners + center + edges)
Repeat for --cam 1. Copy the frames_cam*/ folders back to the Mac.

Assumes Raspberry Pi Camera Module (CSI) via picamera2. If the cameras are USB
webcams instead, see the cv2 fallback note at the bottom.
"""
import argparse
import os

import cv2  # only for imwrite (PNG, lossless)

W, H, FPS = 1280, 720, 100  # confirmed capture mode: 720p @ 100 FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index (0 or 1)")
    ap.add_argument("--out", default="frames", help="output dir")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from picamera2 import Picamera2  # import here so --help works without it

    picam = Picamera2(camera_num=args.cam)
    cfg = picam.create_video_configuration(
        main={"size": (W, H), "format": "RGB888"},
        controls={"FrameRate": FPS},
    )
    picam.configure(cfg)
    picam.start()
    print(f"cam {args.cam} started at {W}x{H}@{FPS}. "
          f"ENTER=save, q+ENTER=quit. Save a pen-UP frame first.")

    i = 0
    while True:
        cmd = input(f"[{i:02d}] ENTER to save (q to quit): ").strip().lower()
        if cmd == "q":
            break
        rgb = picam.capture_array()          # HxWx3 RGB
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # detect.py expects BGR
        path = os.path.join(args.out, f"cam{args.cam}_{i:02d}.png")
        cv2.imwrite(path, bgr)
        print(f"  saved {path}  shape={bgr.shape}")
        i += 1

    picam.stop()
    print(f"done. {i} frames in {args.out}/")


if __name__ == "__main__":
    main()

# ponytail: picamera2 path only. If check_env.sh shows USB cams (/dev/videoN,
# no picamera2), replace the picam block with:
#     cap = cv2.VideoCapture(args.cam)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
#     cap.set(cv2.CAP_PROP_FPS, FPS)
#     ok, bgr = cap.read()   # already BGR, no cvtColor
