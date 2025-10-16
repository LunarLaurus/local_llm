# server/app.py
import logging
import asyncio
import importlib
import os
import sys

LOG = logging.getLogger("laurus-llm")
logging.basicConfig(level=logging.INFO)

# If run as script, fix __package__ for relative imports
if __name__ == "__main__" and __package__ is None:
    LOG.info("Computing package name dynamically based on file location.")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)  # allow imports from parent
    __package__ = "server"  # replace with your package name


from typing import Optional

from fastapi import FastAPI

from config import DEFAULT_MODEL_ID
from taskqueue import init_queue, queue_worker
from generator import Generator  # class (not the singleton)
import generator as generator_module


class LocalLLMServer:
    """
    Encapsulates FastAPI app + generator + queue worker wiring so the whole server can be
    created and started from another project with a single object.\n
        server = LocalLLMServer(bitness="4bit")\n
        server.run(host="127.0.0.1", port=8000)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        bitness: str = "16bit",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        title: str = "Local LLM Server",
        version: str = "2.0",
    ):
        """
        :param model_id: HF model id to load at startup (defaults to DEFAULT_MODEL_ID)
        :param bitness: "4bit", "8bit", or "16bit"
        :param max_tokens: optional default max tokens for generator
        :param temperature: optional default temperature for generator
        """
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.bitness = bitness
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Create the generator instance (but don't load model yet, will load at startup)
        gen_kwargs = {}
        if self.max_tokens is not None:
            gen_kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            gen_kwargs["temperature"] = self.temperature

        # create a local Generator instance (not the module-level singleton yet)
        self.generator = Generator(
            model_id=self.model_id,
            bitness=self.bitness,
            **gen_kwargs,
        )

        # Create the FastAPI app
        self.app = FastAPI(title=title, version=version)

        # Before importing/registering endpoints, set the module-level `generator` object
        # so endpoints that do `from .generator import generator` will bind to our instance
        # upon import.
        generator_module.generator = self.generator

        # Now import endpoints module (fresh) so it picks up the generator instance.
        # Use importlib to ensure standard module import semantics.
        endpoints_mod = importlib.import_module(f"{__package__}.endpoints")
        # endpoints.register_routes will register all routes on the FastAPI app
        endpoints_mod.register_routes(self.app)

        # register startup event on the app that will initialize queue and load model
        @self.app.on_event("startup")
        async def _startup():
            await self._on_startup()

    async def _on_startup(self):
        LOG.info("Server startup (LocalLLMServer)")
        # init queue
        init_queue()

        # start queue worker; pass the generator.generate callable
        # queue_worker will run this callable in an executor
        asyncio.create_task(queue_worker(self.generator.generate))

        # load model in executor (model loading is blocking/heavy)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.generator.load_model, self.model_id)
            LOG.info("Model loaded during startup: %s", self.generator.model_id)
        except Exception:
            LOG.exception("Failed to load model during startup")

    def get_app(self) -> FastAPI:
        """Return the FastAPI app for Uvicorn or mounting."""
        return self.app

    def run(self, host: str = "0.0.0.0", port: int = 8000, **uvicorn_kwargs):
        """
        Convenience: run the server using uvicorn programmatically.
        This is optional; you can also call `uvicorn.run(server.get_app(), ...)` from outside.
        """
        import uvicorn

        LOG.info("Starting uvicorn on %s:%s", host, port)
        uvicorn.run(self.app, host=host, port=port, **uvicorn_kwargs)


if __name__ == "__main__":
    import os

    LLLM_MODEL_ID = (
        os.environ.get("LLLM_MODEL_ID")
        or input(
            "Enter model ID [default: ibm-granite/granite-3b-code-instruct-128k]: "
        )
        or "ibm-granite/granite-3b-code-instruct-128k"
    )

    LLLM_BITNESS = (
        os.environ.get("LLLM_BITNESS")
        or input("Enter quantization bitness (4bit / 8bit / 16bit) [default: 16bit]: ")
        or "16bit"
    )

    LLLM_PORT = int(
        os.environ.get("LLLM_PORT") or input("Enter port [default: 8000]: ") or 8000
    )
    LLLM_HOST = (
        os.environ.get("LLLM_HOST")
        or input("Default host [default: 0.0.0.0]: ")
        or "0.0.0.0"
    )

    print("\n--------------------------------")
    print(f"Starting Local LLM Server")
    print(f"Model:            {LLLM_MODEL_ID}")
    print(f"Host:             {LLLM_HOST}")
    print(f"Port:             {LLLM_PORT}")
    print(f"Quantization:     {LLLM_BITNESS}")
    print("--------------------------------\n")

    # create and run server
    server = LocalLLMServer(model_id=LLLM_MODEL_ID, bitness=LLLM_BITNESS)
    server.run(host=LLLM_HOST, port=LLLM_PORT)
