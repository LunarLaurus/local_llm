# server/app.py
from contextlib import asynccontextmanager
import os
import asyncio
from typing import Optional, cast
from typing import Literal
from fastapi import FastAPI

from .lauruslog import LOG
from .config import DEFAULT_MODEL_ID, MODEL_CHOICES
from .taskqueue import (
    init_queue,
    start_workers,
    stop_workers,
)
from .generator import Generator
from laurus_llm import endpoints


BitnessType = Literal["4bit", "8bit", "16bit"]


# -----------------------------
# CLI helpers
# -----------------------------
def input_with_timeout(
    prompt: str, timeout: int, default: str, env_key: Optional[str] = None
) -> str:
    """
    Prompt user with timeout, returning default if no input,
    or environment variable if set. Shows the default in the prompt.
    """
    # Use environment variable if set
    env = os.environ.get(env_key) if env_key else None
    if env and env.strip():
        return env.strip()

    import threading

    result = [default]

    # Append default to the prompt if not already included
    display_prompt = f"{prompt} [default: {default}]: "

    def _input():
        try:
            user_input = input(display_prompt)
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


def parse_bitness(user_input: str, default: BitnessType = "16bit") -> BitnessType:
    """Convert user input to a valid bitness literal, default if invalid."""
    user_input = user_input.strip().lower()

    mapping = {
        "4": "4bit",
        "4bit": "4bit",
        "8": "8bit",
        "8bit": "8bit",
        "16": "16bit",
        "16bit": "16bit",
    }

    # map or default
    value = mapping.get(user_input, default)

    # cast to satisfy type checker
    return cast(BitnessType, value)


def choose_model() -> str:
    print("\nAvailable models:")
    for idx, model in enumerate(MODEL_CHOICES, start=1):
        print(f"{idx}. {model}")
    print("0. Enter a custom model ID")

    choice = input_with_timeout(
        "Select model by number or type custom: ",
        timeout=30,
        default="1",  # default to first model
        env_key="LLLM_MODEL_ID",
    ).strip()

    # If they typed a number
    if choice.isdigit():
        num = int(choice)
        if num == 0:
            # Ask for custom input
            custom = input_with_timeout(
                "Enter custom model ID: ",
                timeout=30,
                default="ibm-granite/granite-3b-code-instruct-128k",
            ).strip()
            return custom or MODEL_CHOICES[0]
        elif 1 <= num <= len(MODEL_CHOICES):
            return MODEL_CHOICES[num - 1]

    # Otherwise, treat input as custom string
    return choice or MODEL_CHOICES[0]


# -----------------------------
# LocalLLMServer wrapper
# -----------------------------
class LocalLLMServer:
    """Encapsulates FastAPI app, generator, and async task queue."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        bitness: BitnessType = "16bit",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        title: str = "Laurus Local LLM Server",
        version: str = "2.0",
        num_workers: int = 4,
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
        Generator.set_instance(self.generator)

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

    @asynccontextmanager
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
    LLLM_MODEL_ID = choose_model()
    LLLM_BITNESS = parse_bitness(
        input_with_timeout(
            "Enter quantization bitness (4bit / 8bit / 16bit)",
            timeout=10,
            default="16bit",
            env_key="LLLM_BITNESS",
        )
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
