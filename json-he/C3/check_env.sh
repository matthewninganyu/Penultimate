#!/usr/bin/env bash
# Check + install C3 test deps in a local venv, then verify imports.
# Usage:  bash check_env.sh          # check, install if missing, verify
#         bash check_env.sh --test   # also run all three test suites
# Idempotent: safe to re-run. Uses piwheels on Pi automatically (system pip.conf).
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="python3"

command -v "$PY" >/dev/null || { echo "FAIL: python3 not found"; exit 1; }
echo "python3: $($PY --version)"

# Create venv once.
if [ ! -d "$VENV" ]; then
  echo "creating venv $VENV ..."
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Check imports; install only what's missing.
need_install=0
python - <<'PY' || need_install=1
import importlib.util as u
import sys
missing = [m for m in ("numpy", "cv2") if u.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY

if [ "$need_install" -eq 1 ]; then
  echo "installing numpy + opencv-python-headless (piwheels on Pi) ..."
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet numpy opencv-python-headless
fi

# Verify.
python - <<'PY'
import numpy, cv2
print("numpy:", numpy.__version__)
print("cv2  :", cv2.__version__)
print("ENV OK")
PY

# Optional: run the suites.
if [ "${1:-}" = "--test" ]; then
  echo "=== test_detect ==="
  python tests/test_detect.py
  echo "=== A test_position ==="
  python A_triangulation/test_position.py
  echo "=== B test_position ==="
  python B_homography/test_position.py
fi
