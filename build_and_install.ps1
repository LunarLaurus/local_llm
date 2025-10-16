# -------------------------------------
# Windows PowerShell Build & Install Script (No Conda)
# -------------------------------------
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -------------------------------------
# Configuration
# -------------------------------------
$PACKAGE_NAME = "laurus_llm"
$DIST_DIR = "dist"

# -------------------------------------
# Clean old builds
# -------------------------------------
Write-Host "Cleaning old build artifacts..."
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, $DIST_DIR, *.egg-info

# -------------------------------------
# Build new distribution
# -------------------------------------
Write-Host "Building wheel and source distribution..."
pip install --quiet --upgrade pip build wheel setuptools
python -m build --wheel --sdist

# -------------------------------------
# Install wheel
# -------------------------------------
$wheelFiles = @(Get-ChildItem -Path $DIST_DIR -Filter "*.whl" | Sort-Object LastWriteTime -Descending)
if ($wheelFiles.Count -eq 0) {
    Write-Error "Build failed: no wheel file found."
    exit 1
}
$WHEEL_FILE = $wheelFiles[0].FullName

Write-Host "Installing $PACKAGE_NAME from $WHEEL_FILE..."
pip install --upgrade $WHEEL_FILE


# -------------------------------------
# Verify installation
# -------------------------------------
Write-Host "Verifying installation..."
try {
    pip show $PACKAGE_NAME
}
catch {
    Write-Warning "Package info not found."
}

Write-Host ""
Write-Host "$PACKAGE_NAME successfully built and installed."
Write-Host "Installed wheel: $WHEEL_FILE"
