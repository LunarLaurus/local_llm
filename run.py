import os
import subprocess
import sys
import logging

logging.info("Installing application requirements")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
)

logging.info("Running llm_local_server server")
os.execvp(
    "uvicorn",
    ["uvicorn", "llm_local_server:app", "--host", "0.0.0.0", "--port", "8000"],
)
