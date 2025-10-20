# laurus_llm/__init__.py (explicit)
"""laurus_llm — direct exports (eager imports)."""

from .app import LocalLLMServer, input_with_timeout, prompt_bool, parse_bitness
from .endpoints import register_routes
from .generator import Generator
from .models import (
    GenerateRequest,
    GenerateResponse,
    JobResultResponse,
    ModeRequest,
    ReloadRequest,
    ShutdownRequest,
)
from .taskqueue import (
    init_queue,
    enqueue_job,
    start_workers,
    stop_workers,
    _worker_loop,
)
from .lauruslog import LaurusLogger

__all__ = [
    "LocalLLMServer",
    "input_with_timeout",
    "prompt_bool",
    "parse_bitness",
    "register_routes",
    "Generator",
    "GenerateRequest",
    "GenerateResponse",
    "JobResultResponse",
    "ModeRequest",
    "ReloadRequest",
    "ShutdownRequest",
    "init_queue",
    "enqueue_job",
    "start_workers",
    "stop_workers",
    "_worker_loop",
    "LaurusLogger",
]

__version__ = "0.0.3"
