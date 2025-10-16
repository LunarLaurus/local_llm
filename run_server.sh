#!/usr/bin/env bash
set -e

# -------------------------------
# Interactive configuration
# -------------------------------

# Prompt for model
read -p "Enter model ID [default: ibm-granite/granite-3b-code-instruct-128k]: " MODEL_ID
MODEL_ID=${MODEL_ID:-"ibm-granite/granite-3b-code-instruct-128k"}

# Prompt for bitness
read -p "Enter quantization bitness (4bit / 8bit / 16bit) [default: 16bit]: " BITNESS
BITNESS=${BITNESS:-"16bit"}

# Prompt for port
read -p "Enter port [default: 8000]: " PORT
PORT=${PORT:-8000}

# Network config
HOST="0.0.0.0"

# Python module path
MODULE_PATH="laurus_llm.server.app"

# -------------------------------
# Activate Conda environment
# -------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required but not found. Please install Conda first."
    exit 1
fi

# Use environment named after current directory
ENV_NAME=$(basename "$PWD")

if conda env list | grep -qE "^\s*$ENV_NAME\s"; then
    echo "Activating Conda environment: $ENV_NAME..."
    # Use conda's "shell hook" to enable activation in bash scripts
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "Conda environment '$ENV_NAME' not found. Please create it first:"
    echo "  conda create -n $ENV_NAME python=3.12"
    exit 1
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
echo "Conda Env:        $ENV_NAME"
echo "--------------------------------"
echo ""

# -------------------------------
# Run server
# -------------------------------
exec python - <<PYCODE
from laurus_llm.server.app import LocalLLMServer
import uvicorn

server = LocalLLMServer(model_id="${MODEL_ID}", bitness="${BITNESS}")
app = server.get_app()
uvicorn.run(app, host="${HOST}", port=${PORT})
PYCODE