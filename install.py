import subprocess
import sys
import os


def install_package():
    # Ensure pip is up-to-date
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install the current package in editable mode
    package_path = os.path.abspath(".")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", package_path])

    print("Installation complete! You can now import your package.")


if __name__ == "__main__":
    install_package()
