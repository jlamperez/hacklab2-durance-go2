#!/bin/sh
# Create Training/.venv for train_motion_buddy.py.
# The notebook asks for Python 3.10. This uses python3.10 when it is installed.
set -eu
cd "$(dirname "$0")"

if command -v python3.10 >/dev/null 2>&1; then
  PYTHON=python3.10
else
  PYTHON=python3
  echo "python3.10 was not found; using $($PYTHON --version) instead."
fi

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy matplotlib torch h5py

found=0
for wheel in riverops-*.whl; do
  if [ -f "$wheel" ]; then
    .venv/bin/python -m pip install "$wheel"
    found=1
  fi
done
if [ "$found" -eq 0 ]; then
  echo
  echo "riverops is not on PyPI. Put the SDK wheel in this folder and run:"
  echo "  .venv/bin/pip install riverops-*.whl"
fi

echo
echo "Ready. Activate it with: source .venv/bin/activate"
