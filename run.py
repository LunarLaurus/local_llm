import os
import subprocess
import sys
import logging

logging.info("Installing application requirements")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
)

logging.info("Running local_llm_server server")
os.execvp(
    "uvicorn",
    ["uvicorn", "local_llm_server:app", "--host", "0.0.0.0", "--port", "8000"],
)
