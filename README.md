# LED Pen Tracking Prototype

Task 1 detects yellow-orange LED candidates in a still image using OpenCV and NumPy.

## Setup

Use Python 3.12 from a normal Windows Python installation, then create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Detection Only

To generate only the binary mask and candidate debug image:

```powershell
python detect_led.py --image IMG_3298.jpeg
```

In this repository, the sample image is currently under `photos`, so this also works:

```powershell
python detect_led.py --image photos\IMG_3298.jpeg
```

The script writes:

- `output/led_mask.png`
- `output/led_detection.png`

The selected LED uses a temporary image-specific heuristic: among strong candidates, choose the rightmost one as the physical LED instead of the laptop-screen reflection.

## Mark the LED

Run the full prototype with one command:

```powershell
python mark_led.py --image IMG_3298.jpeg
```

The script writes:

- `output/led_mask.png`
- `output/led_marked.png`

It also opens the marked image in an OpenCV window. Press any key while the window is focused to close it.

To mark from an existing mask instead of generating a new one:

```powershell
python mark_led.py --image IMG_3298.jpeg --mask led_mask.png
```

In this repository, bare `IMG_3298.jpeg` resolves from `photos`, and bare `led_mask.png` resolves from `output`.

## Raspberry Pi Realtime Camera

Install the Raspberry Pi camera dependencies:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-venv
```

Create a virtual environment that can see the apt-installed camera packages:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

Verify imports:

```bash
python -c "import cv2; from picamera2 import Picamera2; print('Setup works')"
```

Verify the camera:

```bash
rpicam-hello
```

Run the realtime detector:

```bash
python realtime_led.py --show-mask
```

Useful options:

```bash
python realtime_led.py --width 640 --height 480 --camera 0
python realtime_led.py --min-area 30
python realtime_led.py --headless
```

Stop the detector by pressing `Q` in the preview window. In headless mode, press `Ctrl+C`.

Troubleshooting:

- If the LED is black in the mask, loosen the HSV thresholds in `realtime_led.py`.
- If too much of the image is white, tighten the HSV thresholds.
- If the LED appears but is ignored, reduce `--min-area`.
- If noise is selected, increase `--min-area`.
- If OpenCV windows do not appear over SSH, run on the Raspberry Pi desktop or use `--headless`.
