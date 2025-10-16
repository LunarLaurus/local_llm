#!/usr/bin/env bash
set -e

# -------------------------------------
# Configuration
# -------------------------------------
PACKAGE_NAME="laurus_llm"
DIST_DIR="dist"

# -------------------------------------
# Environment setup
# -------------------------------------
echo "Setting up environment..."

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No virtual environment detected."
    echo "You can create one with: python -m venv .venv && source .venv/bin/activate"
fi

# Ensure build tools are installed
pip install --quiet --upgrade pip build wheel setuptools

# -------------------------------------
# Clean old builds
# -------------------------------------
echo "Cleaning old build artifacts..."
rm -rf build/ "$DIST_DIR"/ *.egg-info

# -------------------------------------
# Build new distribution
# -------------------------------------
echo "Building wheel and source distribution..."
python -m build --wheel --sdist

# -------------------------------------
# Find latest built wheel
# -------------------------------------
WHEEL_FILE=$(ls -t "$DIST_DIR"/*.whl | head -n 1)

if [ -z "$WHEEL_FILE" ]; then
    echo "Build failed: no wheel file found."
    exit 1
fi

# -------------------------------------
# Install (force reinstall)
# -------------------------------------
echo "Installing $PACKAGE_NAME from $WHEEL_FILE..."
pip install --upgrade --force-reinstall "$WHEEL_FILE"

# -------------------------------------
# Verify installation
# -------------------------------------
echo "Verifying installation..."
pip show "$PACKAGE_NAME" || echo "Package info not found."

echo ""
echo "$PACKAGE_NAME successfully built and installed."
echo "Installed wheel: $WHEEL_FILE"