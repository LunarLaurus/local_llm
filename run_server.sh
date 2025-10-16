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

read -p "Enter network address [default: 0.0.0.0]: " LLLM_HOST
LLLM_HOST=${LLLM_HOST:-"0.0.0.0"}

read -p "Enable acess logging? [default: False]: " LLLM_ACCESS_LOG
LLLM_ACCESS_LOG=${LLLM_PORT:-False}
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
echo "Acess Log:        $LLLM_ACCESS_LOG"
echo "--------------------------------"
echo ""

# export variables for the Python script
export LLLM_MODEL_ID LLLM_BITNESS LLLM_PORT LLLM_HOST LLLM_ACCESS_LOG

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#python "$SCRIPT_DIR/laurus_llm/server/app.py"
python -m laurus_llm.server.app