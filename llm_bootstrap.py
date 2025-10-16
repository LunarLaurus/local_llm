# install_and_run.py
import os
import subprocess
import sys

# Install requirements
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
)

# Run server
os.execvp(
    "uvicorn",
    ["uvicorn", "local_llm_server_reload:app", "--host", "0.0.0.0", "--port", "8000"],
)
