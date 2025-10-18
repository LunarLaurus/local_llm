# server/app.py
import os
import asyncio
from typing import Optional
from fastapi import FastAPI

from laurus_llm.lauruslog import LOG
from laurus_llm.config import DEFAULT_MODEL_ID
from laurus_llm.taskqueue import (
    init_queue,
    start_workers,
    stop_workers,
    enqueue_job,
)
from laurus_llm.generator import Generator
import laurus_llm.generator as generator_module
from laurus_llm import endpoints


# -----------------------------
# CLI helpers
# -----------------------------
def input_with_timeout(
    prompt: str, timeout: int, default: str, env_key: Optional[str] = None
) -> str:
    """Prompt with timeout, returning default if no input, or env variable if set."""
    env = os.environ.get(env_key) if env_key else None
    if env and env.strip():
        return env.strip()

    import threading

    result = [default]

    def _input():
        try:
            user_input = input(prompt)
            if user_input.strip():
                result[0] = user_input.strip()
        except Exception:
            pass

    thread = threading.Thread(target=_input, daemon=True)
    thread.start()
    thread.join(timeout)
    return result[0]


def prompt_bool(
    prompt: str, default: bool = False, timeout: int = 10, env_key: Optional[str] = None
) -> bool:
    """Prompt user for yes/no input, supports multiple true/false aliases."""
    default_str = "Y/n" if default else "y/N"
    raw = input_with_timeout(
        f"{prompt} [{default_str}]: ", timeout, str(default), env_key
    )
    raw = raw.strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "t", "true", "1", "ok")


# -----------------------------
# LocalLLMServer wrapper
# -----------------------------
class LocalLLMServer:
    """Encapsulates FastAPI app, generator, and async task queue."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        bitness: str = "16bit",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        title: str = "Local LLM Server",
        version: str = "2.0",
        num_workers: int = 2,
    ):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.bitness = bitness
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_workers = num_workers

        # Generator
        gen_kwargs = {}
        if self.max_tokens is not None:
            gen_kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            gen_kwargs["temperature"] = self.temperature

        self.generator = Generator(
            model_id=self.model_id, bitness=self.bitness, **gen_kwargs
        )
        generator_module.generator = self.generator

        self.worker_tasks_started = False  # track if workers started
        # FastAPI app
        self.app = FastAPI(title=title, version=version, lifespan=self._lifespan)
        endpoints.register_routes(self.app)
        # Always start workers as soon as server instance is created
        asyncio.get_event_loop().create_task(self._start_workers_safe())

    async def _start_workers_safe(self):
        """Start workers if they are not already running."""
        if not self.worker_tasks_started:
            init_queue()
            await start_workers(self.generator.generate, num_workers=self.num_workers)
            self.worker_tasks_started = True

    async def _lifespan(self, app: FastAPI):
        """Startup and shutdown lifecycle for FastAPI."""
        LOG.info("Server startup (lifespan)")
        init_queue()

        # Load model in background thread
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.generator.load_model, self.model_id)
            LOG.info("Model loaded: %s", self.generator.model_id)
        except Exception:
            LOG.exception("Failed to load model during startup")

        # Start async workers
        await start_workers(self.generator.generate, num_workers=self.num_workers)

        yield

        # Shutdown
        LOG.info("Shutting down async workers")
        await stop_workers()

    def get_app(self) -> FastAPI:
        return self.app

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        access_log: bool = False,
        **kwargs,
    ):
        """Run the FastAPI app with uvicorn."""
        import uvicorn

        LOG.info("Starting uvicorn on %s:%s", host, port)
        uvicorn.run(self.app, host=host, port=port, access_log=access_log, **kwargs)


# -----------------------------
# CLI entry point
# -----------------------------
def main():
    LLLM_MODEL_ID = input_with_timeout(
        "Enter model ID",
        timeout=30,
        default="ibm-granite/granite-3b-code-instruct-128k",
        env_key="LLLM_MODEL_ID",
    )
    LLLM_BITNESS = input_with_timeout(
        "Enter quantization bitness (4bit / 8bit / 16bit)",
        timeout=10,
        default="16bit",
        env_key="LLLM_BITNESS",
    )
    LLLM_PORT = int(
        input_with_timeout(
            "Enter port", timeout=10, default="8000", env_key="LLLM_PORT"
        )
    )
    LLLM_HOST = input_with_timeout(
        "Enter host", timeout=10, default="0.0.0.0", env_key="LLLM_HOST"
    )
    LLLM_ACCESS_LOG = prompt_bool(
        "Enable access logging?", default=False, timeout=5, env_key="LLLM_ACCESS_LOG"
    )

    print("\n--------------------------------")
    print(f"Starting Local LLM Server")
    print(f"Model:        {LLLM_MODEL_ID}")
    print(f"Host:         {LLLM_HOST}")
    print(f"Port:         {LLLM_PORT}")
    print(f"Access Log:   {LLLM_ACCESS_LOG}")
    print(f"Quantization: {LLLM_BITNESS}")
    print("--------------------------------\n")

    server = LocalLLMServer(model_id=LLLM_MODEL_ID, bitness=LLLM_BITNESS)
    server.run(host=LLLM_HOST, port=LLLM_PORT, access_log=LLLM_ACCESS_LOG)


if __name__ == "__main__":
    main()
