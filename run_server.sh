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

# -------------------------------------
# Interactive configuration
# -------------------------------------
read -p "Enter model ID [default: ibm-granite/granite-3b-code-instruct-128k]: " MODEL_ID
MODEL_ID=${MODEL_ID:-"ibm-granite/granite-3b-code-instruct-128k"}

read -p "Enter quantization bitness (4bit / 8bit / 16bit) [default: 16bit]: " BITNESS
BITNESS=${BITNESS:-"16bit"}

read -p "Enter port [default: 8000]: " PORT
PORT=${PORT:-8000}

HOST="0.0.0.0"

# -------------------------------------
# Summary
# -------------------------------------
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

# -------------------------------------
# Run server
# -------------------------------------
exec python - <<PYCODE
from laurus_llm.server.app import LocalLLMServer
import uvicorn

server = LocalLLMServer(model_id="${MODEL_ID}", bitness="${BITNESS}")
app = server.get_app()
uvicorn.run(app, host="${HOST}", port=${PORT})
PYCODE
