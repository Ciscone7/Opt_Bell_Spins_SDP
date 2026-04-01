#!/usr/bin/env bash
# setup_env.sh — Create the SpinsSDP virtual environment (macOS / Linux)
# Usage: bash setup_env.sh

set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

echo "=== SpinsSDP environment setup (macOS / Linux) ==="

# Check Python is available
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: '$PYTHON' not found. Install Python 3.10+ and retry."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(sys.version)")
echo "Using Python: $PY_VERSION"

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists — skipping creation."
else
    echo "Creating virtual environment in '$VENV_DIR'..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing runtime dependencies from requirements.txt..."
pip install --prefer-binary -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo "Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
