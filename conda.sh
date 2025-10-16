#!/usr/bin/env bash
set -e

# -------------------------------------
# Common Conda-related variables
# -------------------------------------
CONDA_EXCLUDE_ENV="base"

# -------------------------------------
# Functions
# -------------------------------------

# Check that Conda is installed and callable
check_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: Conda not found. Please install Conda first."
        exit 1
    fi
    
    # Test if conda env list works
    if ! conda env list >/dev/null 2>&1; then
        echo "Error: Failed to list Conda environments. Check your Conda installation."
        exit 1
    fi
}

# Get active or selected Conda environment
select_conda_env() {
    # Use current env if set and not base
    if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "$CONDA_EXCLUDE_ENV" ]; then
        ENV_NAME="$CONDA_DEFAULT_ENV"
        echo "Using current Conda environment: $ENV_NAME"
        return
    fi
    
    # List environments excluding base
    mapfile -t ENV_LIST < <(conda env list \
        | awk 'NR>2 {gsub(/\*/,""); print $1}' \
        | grep -v "^$CONDA_EXCLUDE_ENV$" \
    | grep -v "^$" )
    
    if [ ${#ENV_LIST[@]} -eq 0 ]; then
        echo "No Conda environments found besides $CONDA_EXCLUDE_ENV. Please create one first."
        exit 1
    fi
    
    if [ ${#ENV_LIST[@]} -eq 1 ]; then
        ENV_NAME="${ENV_LIST[0]}"
        echo "Only one non-base Conda environment found. Activating: $ENV_NAME"
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
        return
    fi
    
    # Prompt user to select from multiple environments
    echo "Select a Conda environment to activate (skip $CONDA_EXCLUDE_ENV):"
    select ENV_NAME in "${ENV_LIST[@]}"; do
        if [ -n "$ENV_NAME" ]; then
            echo "Activating Conda environment: $ENV_NAME"
            eval "$(conda shell.bash hook)"
            conda activate "$ENV_NAME"
            break
        else
            echo "Invalid selection."
        fi
    done
}

# Ensure required Python packages are installed in current Conda env
ensure_build_tools() {
    pip install --quiet --upgrade pip build wheel setuptools
}
