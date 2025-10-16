#!/usr/bin/env bash
set -e

# -------------------------------------
# Load common Conda functions
# -------------------------------------
source "$(dirname "$0")/conda.sh"

# -------------------------------------
# Run Conda checks
# -------------------------------------
check_conda
select_conda_env
ensure_build_tools

# -------------------------------------
# Configuration
# -------------------------------------
PACKAGE_NAME="laurus_llm"
DIST_DIR="dist"

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
# Install wheel
# -------------------------------------
WHEEL_FILE=$(ls -t "$DIST_DIR"/*.whl | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo "Build failed: no wheel file found."
    exit 1
fi

echo "Installing $PACKAGE_NAME from $WHEEL_FILE..."
pip install --upgrade "$WHEEL_FILE"

# -------------------------------------
# Verify installation
# -------------------------------------
echo "Verifying installation..."
pip show "$PACKAGE_NAME" || echo "Package info not found."

echo ""
echo "$PACKAGE_NAME successfully built and installed."
echo "Installed wheel: $WHEEL_FILE"
