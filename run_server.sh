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
LLLM_MODEL_ID=${LLLM_MODEL_ID:-"ibm-granite/granite-3b-code-instruct-128k"}

read -p "Enter quantization bitness (4bit / 8bit / 16bit) [default: 16bit]: " BITNESS
LLLM_BITNESS=${LLLM_BITNESS:-"16bit"}

read -p "Enter port [default: 8000]: " LLLM_PORT
LLLM_PORT=${LLLM_PORT:-8000}

LLLM_HOST="0.0.0.0"

# -------------------------------------
# Summary
# -------------------------------------
echo ""
echo "--------------------------------"
echo "Starting Local LLM Server"
echo "Model:            $LLLM_MODEL_ID"
echo "Host:             $LLLM_HOST"
echo "Port:             $LLLM_PORT"
echo "Quantization:     $LLLM_BITNESS"
echo "Conda Env:        $ENV_NAME"
echo "--------------------------------"
echo ""

# export variables for the Python script
export LLLM_MODEL_ID LLLM_BITNESS LLLM_PORT LLLM_HOST

# Run the embedded entrypoint using Python from the Conda env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/server/app.py"
