import asyncio
from typing import Any, Callable
from laurus_llm.lauruslog import LOG

# ---------------- Globals ----------------
job_queue: "asyncio.Queue[dict]" = asyncio.Queue()
jobs: dict[str, dict] = {}
jobs_lock = asyncio.Lock()
worker_tasks: list[asyncio.Task] = []


# ---------------- Queue Management ----------------
def init_queue():
    """Re-initialize the global job queue."""
    global job_queue
    job_queue = asyncio.Queue()
    LOG.info("Job queue re-initialized")


async def enqueue_job(
    job_id: str,
    user_prompt: str,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
):
    """Add a new job to the queue."""
    async with jobs_lock:
        if job_id in jobs:
            raise ValueError(f"Job {job_id} already exists")
        jobs[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
        }

    await job_queue.put(
        {
            "job_id": job_id,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    )
    LOG.info("Enqueued job %s", job_id)


# ---------------- Worker Management ----------------
async def start_workers(generator_fn: Callable[..., Any], num_workers: int = 1):
    """Start multiple worker tasks."""
    for i in range(num_workers):
        task = asyncio.create_task(_worker_loop(generator_fn))
        worker_tasks.append(task)
        LOG.info("Worker %d started", i + 1)


async def stop_workers():
    """Cancel all running worker tasks."""
    LOG.info("Stopping all workers...")
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    worker_tasks.clear()
    LOG.info("All workers stopped")


# ---------------- Worker Loop ----------------
async def _worker_loop(generator_fn: Callable[..., Any]):
    """Worker loop that continuously processes jobs."""
    LOG.info("Worker loop started")
    while True:
        job = await job_queue.get()
        job_id = job.get("job_id")

        if not job_id:
            LOG.error("Malformed job: %s", job)
            job_queue.task_done()
            continue

        async with jobs_lock:
            j = jobs.get(job_id)
            if not j:
                LOG.warning("Job %s not found; skipping", job_id)
                job_queue.task_done()
                continue
            if j.get("status") == "cancelled":
                LOG.info("Job %s cancelled; skipping", job_id)
                job_queue.task_done()
                continue
            j["status"] = "running"

        try:
            # Run generator function (async or sync)
            if asyncio.iscoroutinefunction(generator_fn):
                result = await generator_fn(
                    job["user_prompt"],
                    job.get("system_prompt"),
                    job.get("max_tokens"),
                    job.get("temperature"),
                )
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    generator_fn,
                    job["user_prompt"],
                    job.get("system_prompt"),
                    job.get("max_tokens"),
                    job.get("temperature"),
                )

            async with jobs_lock:
                if j.get("status") != "cancelled":
                    j["status"] = "done"
                    j["result"] = result
                    LOG.info("Job %s completed successfully", job_id)
                else:
                    LOG.info("Job %s cancelled after execution", job_id)

        except Exception as e:
            LOG.exception("Job %s failed", job_id)
            async with jobs_lock:
                j["status"] = "error"
                j["error"] = str(e)
        finally:
            job_queue.task_done()
