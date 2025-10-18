import os
import yaml


def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


cfg = load_config()

DEFAULT_MODEL_ID = cfg.get("local_model", "ibm-granite/granite-3b-code-instruct-128k")
DEFAULT_MAX_TOKENS = int(cfg.get("default_max_tokens", 512))
DEFAULT_TEMP = float(cfg.get("default_temperature", 0.2))

MODES = {
    "c": "Concise summarizer for C source code.",
    "asm": "Concise summarizer for assembly code.",
    "file": "File-level summarizer for code.",
    "python": "Concise summarizer for Python code.",
    "java": "Concise summarizer for Java code.",
    "cpp": "Concise summarizer for C++ code.",
}

MODEL_CHOICES = [
    # Core large models
    "ibm-granite/granite-3b-code-instruct-128k",
    "ibm-granite/granite-8b-code-instruct-128k",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/wavecoder-ds-6.7b",
    "ise-uiuc/Magicoder-S-DS-6.7B",
    # Compact chat/code models
    "StabilityAI/stable-code-instruct-3b",
    "microsoft/Phi-3-mini-128k-instruct",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-3B-Instruct",
]
