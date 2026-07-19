# Penultimate

Penultimate turns a normal laptop screen into a writable surface using a Raspberry Pi, two Raspberry Pi Camera Modules, Picamera2, OpenCV, NumPy, stereo triangulation, and a pen whose bright yellow/orange LED is located directly at the physical pen tip.

The final runtime estimates screen-relative pen coordinates, normalized coordinates, laptop pixel coordinates, distance from the screen, hover/touch state, tracking confidence, contact confidence, timestamps, and sequence numbers. It does not move the operating-system cursor or draw yet.

## Hardware Assumptions

- The LED is constantly illuminated and sits directly at the physical writing tip.
- There is no LED-to-tip offset.
- The screen is reflective: hover often shows a physical LED blob plus a reflection/glare blob.
- Stereo geometry estimates position; reflection/glare evidence estimates contact.
- The cameras are detachable. Run guided screen calibration every time either camera moves.
- Camera intrinsic calibration is separate and only needs repeating when the camera module, lens/focus, runtime resolution, or optical setup changes.

## Raspberry Pi Setup

Install packages:

```bash
sudo apt update
sudo apt install -y \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-venv
```

Create a virtual environment that can see the apt-installed camera packages:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

Verify imports:

```bash
python -c "import cv2, numpy; from picamera2 import Picamera2; print('Setup works')"
```

Confirm both cameras are detected:

```bash
rpicam-hello --list-cameras
```

Do not use `cv2.VideoCapture` for Raspberry Pi CSI cameras. The project uses Picamera2.

## Calibration

Calibrate intrinsics once per camera at the same resolution used for runtime:

```bash
python calibrate_intrinsics.py \
    --camera 0 \
    --output calibration/camera_0_intrinsics.npz

python calibrate_intrinsics.py \
    --camera 1 \
    --output calibration/camera_1_intrinsics.npz
```

Use a checkerboard with varied positions, rotations, distances, and angles. Press `SPACE` to capture a valid sample and `Q` to quit.

Run guided screen-pose calibration after moving either camera:

```bash
python calibrate_screen.py \
    --screen-width-mm 344.0 \
    --screen-height-mm 194.0 \
    --screen-width-px 1920 \
    --screen-height-px 1080 \
    --show-mask
```

The screen coordinate system is top-left origin, +X right, +Y down, +Z outward from the display toward the user. The guided four-point order is top-left, top-right, bottom-right, bottom-left. Add `--nine-point` for extra validation.

During screen calibration, touch each instructed location with the LED tip and hold still. The program collects stable samples from both cameras, estimates each camera pose with planar PnP, builds projection matrices, validates reprojection error, and writes `calibration/screen_calibration.npz`.

For the realtime homography mapper, run the 9-target calibration:

```bash
python calibrate_homography.py \
    --screen-width-px 1920 \
    --screen-height-px 1080 \
    --show-mask
```

This collects top/middle/bottom targets instead of only four corners. By default, camera 0 skips the top-left target and camera 1 skips the top-right target, which avoids training on oversized LED blobs near the physical cameras. It writes `calibration/screen_homography.npz`.

Optional: write a screen-sized target image first:

```bash
python calibrate_homography.py \
    --screen-width-px 1920 \
    --screen-height-px 1080 \
    --target-image output/homography_targets_9.png \
    --write-target-only
```

## Runtime

Primary calibrated preview:

```bash
python realtime.py --show-mask
```

Networked calibrated runtime:

```bash
python realtime.py \
    --show-mask \
    --send-udp \
    --laptop-ip 192.168.50.1 \
    --laptop-port 5005
```

Headless sender:

```bash
python realtime.py \
    --headless \
    --send-udp \
    --laptop-ip 192.168.50.1 \
    --laptop-port 5005
```

Uncalibrated LED preview for camera/debug work:

```bash
python realtime.py --preview-only --show-mask
```

Offline image-pair debug mode:

```bash
python realtime.py \
    --offline-left tests/images/camera_0.png \
    --offline-right tests/images/camera_1.png \
    --preview-only \
    --show-mask
```

Useful tuning flags:

```bash
python realtime.py --min-area 30
python realtime.py --lower-h 10 --lower-s 150 --lower-v 220 --upper-h 40 --upper-s 255 --upper-v 255
python realtime.py --max-reprojection-error 8 --max-frame-skew-ms 20
python realtime.py --smoothing-alpha 0.45 --tracking-confidence-threshold 0.25
```

Runtime keys:

- `Q`: quit
- `S`: save debug snapshot frames and masks under `output/`
- `D`: toggle detailed overlays
- `C`: print a recalibration reminder

## UDP Receiver

Run this on the laptop:

```bash
python pen_receiver.py --port 5005
```

Optional CSV logging:

```bash
python pen_receiver.py --port 5005 --csv output/pen_packets.csv
```

The receiver validates required fields, ignores malformed packets, rejects old or duplicate sequence numbers, prints the latest finalized pen state, treats packet timeout as tracking lost, and never leaves touch active after communication stops.

## Output Packet

Each finalized packet contains:

```json
{
  "sequence": 1524,
  "timestamp": 1784389912.184,
  "valid": true,
  "normalized_x": 0.437,
  "normalized_y": 0.681,
  "pixel_x": 839,
  "pixel_y": 735,
  "x_mm": 150.3,
  "y_mm": 132.1,
  "distance_mm": 2.8,
  "touching": true,
  "contact_confidence": 0.91,
  "tracking_confidence": 0.94,
  "frame_skew_ms": 2.4
}
```

When invalid, position fields are `null`, `valid` is `false`, and `touching` is `false`.

## Troubleshooting

- If either camera is missing, run `rpicam-hello --list-cameras`, check ribbons/connectors, and confirm `Picamera2.global_camera_info()` sees two modules.
- If the LED is black in the mask, lower `--lower-v` or saturation thresholds.
- If the mask is mostly white, raise `--lower-s`, raise `--lower-v`, or narrow hue bounds.
- If dim noise is selected, increase `--min-area`.
- If the physical LED/reflection choice is wrong, inspect the detailed overlay and check reprojection error; stereo geometry should beat left/right heuristics after screen calibration.
- If reprojection error is high, rerun `calibrate_screen.py` with steadier corner touches and confirm intrinsics match the runtime resolution.
- If triangulation is unstable, reduce pen speed, check frame skew, improve lighting, and ensure both camera views see the same LED.
- If contact flickers, tune contact thresholds in `config.py`; contact uses both blob separation/merging and triangulated distance with hysteresis.
- If OpenCV windows do not appear over SSH, run from the Raspberry Pi desktop or use `--headless`.
- Stop programs with `Q` in preview windows or `Ctrl+C` in the terminal; cleanup attempts to stop both cameras, close sockets, and close OpenCV windows.

## Local Tests

Tests avoid physical camera dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Static-image prototype scripts (`detect_led.py`, `mark_led.py`, `realtime_led.py`) remain in the repository for reference and are not required by the calibrated runtime.
