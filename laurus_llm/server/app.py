import asyncio
from typing import Optional

from fastapi import FastAPI

from laurus_llm.lauruslog import LOG
from laurus_llm.server.config import DEFAULT_MODEL_ID, MODEL_CHOICES
from laurus_llm.server.taskqueue import init_queue, queue_worker
from laurus_llm.server.generator import Generator
import laurus_llm.server.generator as generator_module


class LocalLLMServer:
    """Encapsulates FastAPI app + generator + queue worker."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        bitness: str = "16bit",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        title: str = "Local LLM Server",
        version: str = "2.0",
    ):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.bitness = bitness
        self.max_tokens = max_tokens
        self.temperature = temperature

        gen_kwargs = {}
        if self.max_tokens is not None:
            gen_kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            gen_kwargs["temperature"] = self.temperature

        # Create generator instance
        self.generator = Generator(
            model_id=self.model_id,
            bitness=self.bitness,
            **gen_kwargs,
        )

        # Create FastAPI app
        self.app = FastAPI(title=title, version=version)

        # Bind generator to module-level object for endpoints
        generator_module.generator = self.generator

        # Import endpoints and register
        from laurus_llm.server import endpoints

        endpoints.register_routes(self.app)

        # Startup event: init queue + load model
        @self.app.on_event("startup")
        async def _startup():
            await self._on_startup()

    async def _on_startup(self):
        LOG.info("Server startup (LocalLLMServer)")
        init_queue()
        asyncio.create_task(queue_worker(self.generator.generate))
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.generator.load_model, self.model_id)
            LOG.info("Model loaded: %s", self.generator.model_id)
        except Exception:
            LOG.exception("Failed to load model during startup")

    def get_app(self) -> FastAPI:
        return self.app

    def run(self, host: str = "0.0.0.0", port: int = 8000, **uvicorn_kwargs):
        """Run the server via Uvicorn."""
        import uvicorn

        LOG.info("Starting uvicorn on %s:%s", host, port)
        uvicorn.run(self.app, access_log=False, host=host, port=port, **uvicorn_kwargs)


# -----------------------------
# CLI entry point
# -----------------------------
def main():
    import os

    LLLM_MODEL_ID = os.environ.get("LLLM_MODEL_ID") or DEFAULT_MODEL_ID
    LLLM_BITNESS = os.environ.get("LLLM_BITNESS") or "16bit"
    LLLM_HOST = os.environ.get("LLLM_HOST") or "0.0.0.0"
    LLLM_PORT = int(os.environ.get("LLLM_PORT") or 8000)

    print("\n--------------------------------")
    print(f"Starting Local LLM Server")
    print(f"Model:        {LLLM_MODEL_ID}")
    print(f"Host:         {LLLM_HOST}")
    print(f"Port:         {LLLM_PORT}")
    print(f"Quantization: {LLLM_BITNESS}")
    print("--------------------------------\n")

    server = LocalLLMServer(model_id=LLLM_MODEL_ID, bitness=LLLM_BITNESS)
    server.run(host=LLLM_HOST, port=LLLM_PORT)


if __name__ == "__main__":
    main()
