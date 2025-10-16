#!/usr/bin/env bash
set -e

# -------------------------------
# Interactive configuration
# -------------------------------

# Prompt for model
read -p "Enter model ID [default: ibm-granite/granite-3b-code-instruct-128k]: " MODEL_ID
MODEL_ID=${MODEL_ID:-"ibm-granite/granite-3b-code-instruct-128k"}

# Prompt for bitness
read -p "Enter quantization bitness (4bit / 8bit / 16bit) [default: 4bit]: " BITNESS
BITNESS=${BITNESS:-"16bit"}

# Prompt for port
read -p "Enter port [default: 8000]: " PORT
PORT=${PORT:-8000}

# Network config
HOST="0.0.0.0"

# Python module path
MODULE_PATH="laurus_local_llm.server.app"

# -------------------------------
# Optional: activate venv
# -------------------------------
if [ -d ".venv" ]; then
    source .venv/bin/activate
    elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# -------------------------------
# Summary
# -------------------------------
echo ""
echo "--------------------------------"
echo "Starting Local LLM Server"
echo "Model:            $MODEL_ID"
echo "Host:             $HOST"
echo "Port:             $PORT"
echo "Quantization:     $BITNESS"
echo "--------------------------------"
echo ""

# -------------------------------
# Run server
# -------------------------------
exec python - <<PYCODE
from laurus_local_llm.server.app import LocalLLMServer
import uvicorn

server = LocalLLMServer(model_id="${MODEL_ID}", bitness="${BITNESS}")
app = server.get_app()
uvicorn.run(app, host="${HOST}", port=${PORT})
PYCODE
