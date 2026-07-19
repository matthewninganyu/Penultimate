#!/usr/bin/env bash
# Check + install ALL C3 deps on system python3 (tests AND live viewer).
# One environment: apt system packages (picamera2 can't come from pip; the live
# window needs GUI opencv, not headless). No venv.
# Usage:  bash check_env.sh          # check, apt-install missing, verify
#         bash check_env.sh --test   # also run the three test suites
set -euo pipefail
cd "$(dirname "$0")"

command -v python3 >/dev/null || { echo "FAIL: python3 not found"; exit 1; }
echo "python3: $(python3 --version)"

# Which imports are missing? (picamera2 is Pi-only; needed for live_dual, not tests)
missing=$(python3 - <<'PY'
import importlib.util as u
mods = {"numpy":"python3-numpy", "cv2":"python3-opencv",
        "matplotlib":"python3-matplotlib", "picamera2":"python3-picamera2"}
print(" ".join(pkg for m, pkg in mods.items() if u.find_spec(m) is None))
PY
)

if [ -n "$missing" ]; then
  echo "missing -> apt packages:$missing"
  if command -v apt-get >/dev/null; then
    sudo apt-get update
    # shellcheck disable=SC2086
    sudo apt-get install -y $missing
  else
    echo "no apt (not a Pi/Debian?). On Mac dev: pip install numpy opencv-python matplotlib"
    echo "(picamera2 is Pi-only — live_dual.py runs on the Pi.)"
    exit 1
  fi
fi

# Verify (picamera2 optional — absent is fine off-Pi).
python3 - <<'PY'
import numpy, cv2
print("numpy      :", numpy.__version__)
print("cv2        :", cv2.__version__)
try:
    import matplotlib; print("matplotlib :", matplotlib.__version__)
except Exception as e:
    print("matplotlib : MISSING", e)
try:
    import picamera2; print("picamera2  : ok (live_dual ready)")
except Exception:
    print("picamera2  : absent (fine off-Pi; needed for live_dual.py)")
print("ENV OK")
PY

if [ "${1:-}" = "--test" ]; then
  echo "=== test_detect ===";      python3 tests/test_detect.py
  echo "=== A test_position ==="; python3 A_triangulation/test_position.py
  echo "=== A test_calibrate ==="; python3 A_triangulation/test_calibrate.py
fi
