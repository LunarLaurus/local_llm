# server/taskqueue.py
import asyncio
import logging
from typing import Callable, Any

LOG = logging.getLogger("laurus-llm")

job_queue: "asyncio.Queue[dict]" = None
jobs: dict = {}
jobs_lock = asyncio.Lock()
gen_lock = asyncio.Lock()


def init_queue():
    """Initialize the global job_queue. Call at startup."""
    global job_queue
    job_queue = asyncio.Queue()


async def queue_worker(generator_fn: Callable[..., str]):
    """
    Worker that consumes jobs from job_queue and runs generator_fn in an executor.

    generator_fn is expected to be a synchronous function with signature:
        generator_fn(user_prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> str

    The worker will call it via run_in_executor to avoid blocking the event loop.
    """
    if job_queue is None:
        raise RuntimeError(
            "job_queue not initialized. Call init_queue() before starting the worker."
        )

    LOG.info("Queue worker started")
    while True:
        job = await job_queue.get()
        try:
            job_id = job["job_id"]
        except Exception:
            LOG.error("Malformed job retrieved from queue: %s", job)
            job_queue.task_done()
            continue

        # mark job running (if exists)
        async with jobs_lock:
            j = jobs.get(job_id)
            if not j:
                LOG.warning("Job %s not found in jobs store; skipping", job_id)
                job_queue.task_done()
                continue
            if j.get("status") == "cancelled":
                LOG.info("Job %s already cancelled; skipping", job_id)
                job_queue.task_done()
                continue
            j["status"] = "running"

        try:
            loop = asyncio.get_running_loop()
            # NOTE: call generator_fn with (user_prompt, system_prompt, max_tokens, temperature)
            result_text = await loop.run_in_executor(
                None,
                generator_fn,
                job["user_prompt"],
                job["system_prompt"],
                job["max_tokens"],
                job["temperature"],
            )

            async with jobs_lock:
                if jobs[job_id].get("status") != "cancelled":
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["result"] = result_text
                    LOG.info("Job %s completed", job_id)
                else:
                    LOG.info("Job %s was cancelled after running", job_id)

        except Exception as e:
            LOG.exception("Job %s failed during generation", job_id)
            async with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)

        finally:
            job_queue.task_done()
