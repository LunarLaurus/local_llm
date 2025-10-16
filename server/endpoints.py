# server/endpoints.py
import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from .models import (
    GenerateRequest,
    GenerateResponse,
    JobResultResponse,
    ModeRequest,
    ReloadRequest,
    ShutdownRequest,
)
from .queue import jobs, jobs_lock, job_queue
from .config import MODES, DEFAULT_MAX_TOKENS, DEFAULT_TEMP
from .generator import generator  # singleton Generator instance

LOG = logging.getLogger("laurus-llm")


def register_routes(app: FastAPI):
    @app.post("/generate", response_model=GenerateResponse)
    async def enqueue_generate(req: GenerateRequest):
        if not req.user_prompt:
            raise HTTPException(400, "user_prompt required")

        # read mode from generator
        system_prompt = req.system_prompt or generator.current_mode.get("system_prompt")
        max_tokens = req.max_tokens or DEFAULT_MAX_TOKENS
        temperature = req.temperature or DEFAULT_TEMP

        job_id = str(uuid.uuid4())
        job_entry = {
            "job_id": job_id,
            "status": "pending",
            "system_prompt": system_prompt,
            "user_prompt": req.user_prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "result": None,
            "error": None,
        }
        async with jobs_lock:
            jobs[job_id] = job_entry
        await job_queue.put(job_entry)
        return GenerateResponse(job_id=job_id)

    @app.get("/result/{job_id}", response_model=JobResultResponse)
    async def get_result(job_id: str):
        async with jobs_lock:
            j = jobs.get(job_id)
        if not j:
            raise HTTPException(404, "job_id not found")
        return JobResultResponse(
            job_id=job_id, status=j["status"], result=j["result"], error=j["error"]
        )

    @app.post("/cancel/{job_id}")
    async def cancel_job(job_id: str):
        async with jobs_lock:
            if job_id not in jobs:
                raise HTTPException(404, "job_id not found")
            if jobs[job_id]["status"] in ("done", "error"):
                raise HTTPException(400, "Cannot cancel completed job")
            jobs[job_id]["status"] = "cancelled"
        return {"status": "cancelled", "job_id": job_id}

    @app.post("/mode")
    async def set_mode(req: ModeRequest):
        mode = req.mode.lower()
        if mode not in list(MODES.keys()) + ["custom"]:
            raise HTTPException(400, "invalid mode")
        if mode == "custom":
            if not req.custom_system_prompt:
                raise HTTPException(400, "custom_system_prompt required")
            # update generator's current_mode
            generator.current_mode["name"] = "custom"
            generator.current_mode["system_prompt"] = req.custom_system_prompt
        else:
            generator.current_mode["name"] = mode
            generator.current_mode["system_prompt"] = MODES[mode]
        return {
            "mode": generator.current_mode["name"],
            "system_prompt": generator.current_mode["system_prompt"],
        }

    @app.post("/reload")
    async def reload_model_endpoint(req: ReloadRequest):
        model_id = req.model_id or generator.model_id
        # generator.load_model is synchronous in the provided design.
        # If loading is heavy/blocking, run it in an executor:
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, generator.load_model, model_id)
        return {"status": "reloaded", "model_id": generator.model_id}

    @app.post("/shutdown")
    async def shutdown(req: ShutdownRequest, background_tasks: BackgroundTasks):
        LOG.info("Shutdown requested: %s", req.reason)

        # schedule immediate exit in background
        def _exit():
            os._exit(0)

        background_tasks.add_task(_exit)
        return {"status": "shutting down", "reason": req.reason}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": generator.pipeline is not None,
            "mode": generator.current_mode.get("name"),
            "queue_size": job_queue.qsize() if job_queue else None,
            "model_id": generator.model_id,
        }
