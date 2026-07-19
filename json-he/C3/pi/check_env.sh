#!/usr/bin/env bash
# Run ON the Raspberry Pi. Reports what's present and what to install.
# Usage: bash check_env.sh
set -u

echo "=== System ==="
uname -a
cat /etc/os-release 2>/dev/null | grep PRETTY_NAME
echo

echo "=== Python ==="
python3 --version || echo "MISSING: python3"
echo

echo "=== Python packages ==="
check_py () {  # $1 = import name, $2 = pip/apt hint
  if python3 -c "import $1" 2>/dev/null; then
    ver=$(python3 -c "import $1; print(getattr($1,'__version__','?'))" 2>/dev/null)
    echo "OK   $1 ($ver)"
  else
    echo "MISS $1   -> install: $2"
  fi
}
check_py numpy      "pip3 install numpy"
check_py cv2        "sudo apt install -y python3-opencv   (or pip3 install opencv-python-headless)"
check_py picamera2  "sudo apt install -y python3-picamera2"
echo

echo "=== Camera stack ==="
if command -v rpicam-hello >/dev/null 2>&1; then
  echo "OK   rpicam-hello present"; rpicam-hello --list-cameras 2>&1 | sed 's/^/     /'
elif command -v libcamera-hello >/dev/null 2>&1; then
  echo "OK   libcamera-hello present"; libcamera-hello --list-cameras 2>&1 | sed 's/^/     /'
else
  echo "MISS libcamera/rpicam CLI -> sudo apt install -y rpicam-apps (or libcamera-apps)"
fi
echo
echo "USB webcams (if cameras are USB, not CSI):"
ls /dev/video* 2>/dev/null || echo "  none at /dev/video*"
echo
echo "=== Done. Fix any MISS above, then run capture.py ==="
