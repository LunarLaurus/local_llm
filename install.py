import subprocess
import sys
import os
import logging


def install_package():

    logging.info("Ensure pip is up-to-date")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    logging.info("Install the current package in editable mode")
    package_path = os.path.abspath(".")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", package_path])

    logging.info("Installation complete! You can now import your package.")


if __name__ == "__main__":
    install_package()
