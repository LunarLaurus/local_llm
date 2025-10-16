#!/usr/bin/env bash
set -e

# -------------------------------
# Interactive configuration
# -------------------------------

read -p "Enter model ID [default: ibm-granite/granite-3b-code-instruct-128k]: " MODEL_ID
MODEL_ID=${MODEL_ID:-"ibm-granite/granite-3b-code-instruct-128k"}

read -p "Enter quantization bitness (4bit / 8bit / 16bit) [default: 16bit]: " BITNESS
BITNESS=${BITNESS:-"16bit"}

read -p "Enter port [default: 8000]: " PORT
PORT=${PORT:-8000}

HOST="0.0.0.0"
MODULE_PATH="laurus_llm.server.app"

# -------------------------------
# Activate Conda environment
# -------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required but not found. Please install Conda first."
    exit 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# If no environment active or base is active, prompt user
CURRENT_ENV=$(conda info --json | jq -r '.active_prefix_name')

if [[ -z "$CURRENT_ENV" || "$CURRENT_ENV" == "base" ]]; then
    # List environments, skip base
    echo "Select a Conda environment to activate:"
    mapfile -t ENV_LIST < <(conda env list | awk '{print $1}' | grep -vE '^(#|base)$')
    
    if [[ ${#ENV_LIST[@]} -eq 0 ]]; then
        echo "No Conda environments found. Please create one first."
        exit 1
    fi
    
    for i in "${!ENV_LIST[@]}"; do
        echo "[$i] ${ENV_LIST[$i]}"
    done
    
    read -p "Enter number of environment to activate: " ENV_IDX
    
    if [[ -z "${ENV_LIST[$ENV_IDX]}" ]]; then
        echo "Invalid selection."
        exit 1
    fi
    
    ENV_NAME="${ENV_LIST[$ENV_IDX]}"
    echo "Activating Conda environment: $ENV_NAME..."
    conda activate "$ENV_NAME"
else
    ENV_NAME="$CURRENT_ENV"
    echo "Using active Conda environment: $ENV_NAME"
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
