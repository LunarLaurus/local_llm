# laurus_llm/server/__init__.py
"""
Server package public API.

Exports:
- Generator: the model/pipeline class
- generator: optional singleton instance (if present)
- init_queue, job_queue, jobs, jobs_lock, queue_worker: queue primitives
- register_routes: function to add endpoints to a FastAPI app
- models: Pydantic request/response models
- config defaults
- LocalLLMServer: helper class to create/run the full server
"""

# core components
from generator import Generator

# generator singleton may or may not exist in your package; import safely
try:
    from generator import generator
except Exception:
    generator = None

# queue primitives
from taskqueue import init_queue, job_queue, jobs, jobs_lock, queue_worker

# models
from models import (
    GenerateRequest,
    GenerateResponse,
    JobResultResponse,
    ModeRequest,
    ReloadRequest,
    ShutdownRequest,
)

# config / defaults
from config import DEFAULT_MODEL_ID, DEFAULT_MAX_TOKENS, DEFAULT_TEMP, MODES

# endpoints helper (registers routes on a FastAPI app)
from endpoints import register_routes

# app helper class for easy programmatic setup
from app import LocalLLMServer

__all__ = [
    "Generator",
    "generator",
    "init_queue",
    "job_queue",
    "jobs",
    "jobs_lock",
    "queue_worker",
    "GenerateRequest",
    "GenerateResponse",
    "JobResultResponse",
    "ModeRequest",
    "ReloadRequest",
    "ShutdownRequest",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMP",
    "MODES",
    "register_routes",
    "LocalLLMServer",
]
