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
