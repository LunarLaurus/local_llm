#!/usr/bin/env bash
set -e

PACKAGE_NAME="laurus_llm"
DIST_DIR="dist"

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Ensure build tools
pip install --quiet --upgrade pip build wheel setuptools

# Clean old builds
rm -rf build/ "$DIST_DIR"/ *.egg-info

# Build
python -m build --wheel --sdist

# Latest wheel
WHEEL_FILE=$(ls -t "$DIST_DIR"/*.whl | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo "Build failed: no wheel file found."
    exit 1
fi

# Only install if not installed or wheel is newer
INSTALLED_VERSION=$(pip show "$PACKAGE_NAME" | grep Version | awk '{print $2}' || echo "")
WHEEL_VERSION=$(basename "$WHEEL_FILE" | sed -E "s/^${PACKAGE_NAME}-([0-9\.]+)-.*$/\1/")

if [ "$INSTALLED_VERSION" != "$WHEEL_VERSION" ]; then
    echo "Installing $PACKAGE_NAME version $WHEEL_VERSION..."
    pip install --upgrade "$WHEEL_FILE"
else
    echo "$PACKAGE_NAME version $INSTALLED_VERSION is up-to-date. Skipping install."
fi

pip show "$PACKAGE_NAME"
