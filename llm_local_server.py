# local_llm_server_reload_v2.py
"""
FastAPI LLM server (improved):
- job queue (unbounded)
- /generate, /result/{job_id}, /cancel/{job_id}, /mode, /shutdown
- /reload: load a new HF model dynamically
"""
import os, uuid, yaml, logging, asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)

LOG = logging.getLogger("local_llm_server_reload_v2")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Local LLM Server (Reloadable)", version="2.0")


# ---------------- Config & Defaults ----------------
def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


cfg = load_config()
DEFAULT_MODEL_ID = cfg.get("local_model", "mistralai/Mistral-7B-Instruct-v0.2")
DEFAULT_MAX_TOKENS = int(cfg.get("default_max_tokens", 512))
DEFAULT_TEMP = float(cfg.get("default_temperature", 0.2))

# ---------------- Globals ----------------
tokenizer = None
model = None
generator = None
current_model_id = DEFAULT_MODEL_ID

job_queue: "asyncio.Queue[dict]" = None
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = asyncio.Lock()
gen_lock = asyncio.Lock()

# ---------------- Supported modes ----------------
MODES = {
    "c": "Concise summarizer for C source code. Focus on function purpose, inputs/outputs, types, complexity.",
    "asm": "Concise summarizer for assembly code. Explain registers, side-effects, high-level constructs.",
    "file": "File-level summarizer for code. Overview file content, key functions, dependencies, risks.",
    "python": "Concise summarizer for Python code. Explain function behavior, inputs, outputs, exceptions.",
    "java": "Concise summarizer for Java code. Focus on class/method purpose, parameters, return values.",
    "cpp": "Concise summarizer for C++ code. Highlight functions, types, complexity, and side-effects.",
}
current_mode = {"name": "c", "system_prompt": MODES["c"]}


# ---------------- Pydantic models ----------------
class GenerateRequest(BaseModel):
    user_prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str = "queued"


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None


class ModeRequest(BaseModel):
    mode: str
    custom_system_prompt: Optional[str] = None


class ReloadRequest(BaseModel):
    model_id: Optional[str] = None


class ShutdownRequest(BaseModel):
    reason: Optional[str] = "shutdown requested"


# ---------------- Startup ----------------
@app.on_event("startup")
async def startup_event():
    global tokenizer, model, generator, job_queue, current_model_id
    LOG.info("Server startup: loading model %s", current_model_id)
    await load_model(current_model_id)
    job_queue = asyncio.Queue()  # unbounded queue
    asyncio.create_task(queue_worker())


async def load_model(model_id: str):
    global tokenizer, model, generator, current_model_id
    LOG.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    try:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        LOG.info("Loaded 4-bit quantized model")
    except Exception:
        LOG.warning("4-bit load failed, falling back to float16")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device_map="auto",
    )
    current_model_id = model_id
    LOG.info("Model pipeline ready")


# ---------------- Queue worker ----------------
async def queue_worker():
    LOG.info("Queue worker running")
    while True:
        job = await job_queue.get()
        job_id = job["job_id"]
        async with jobs_lock:
            if jobs[job_id].get("status") == "cancelled":
                job_queue.task_done()
                continue
            jobs[job_id]["status"] = "running"
        try:
            loop = asyncio.get_running_loop()
            result_text = await loop.run_in_executor(
                None,
                run_generation_sync,
                job["system_prompt"],
                job["user_prompt"],
                job["max_tokens"],
                job["temperature"],
            )
            async with jobs_lock:
                if jobs[job_id]["status"] != "cancelled":
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["result"] = result_text
        except Exception as e:
            LOG.exception("Job %s failed", job_id)
            async with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
        finally:
            job_queue.task_done()


def run_generation_sync(
    system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
) -> str:
    full_prompt = f"{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:"
    max_new_tokens = max(1, min(2000, int(max_tokens)))
    temp = float(temperature)
    if not hasattr(run_generation_sync, "_thread_lock"):
        import threading

        run_generation_sync._thread_lock = threading.Lock()
    with run_generation_sync._thread_lock:
        outputs = generator(
            full_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=(temp > 0),
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = outputs[0].get("generated_text", "")
    if "Assistant:" in text:
        return text.split("Assistant:")[-1].strip()
    return text.replace(full_prompt, "").strip()


# ---------------- Endpoints ----------------
@app.post("/generate", response_model=GenerateResponse)
async def enqueue_generate(req: GenerateRequest):
    if not req.user_prompt:
        raise HTTPException(400, "user_prompt required")
    system_prompt = req.system_prompt or current_mode["system_prompt"]
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
        current_mode["name"] = "custom"
        current_mode["system_prompt"] = req.custom_system_prompt
    else:
        current_mode["name"] = mode
        current_mode["system_prompt"] = MODES[mode]
    return {
        "mode": current_mode["name"],
        "system_prompt": current_mode["system_prompt"],
    }


@app.post("/reload")
async def reload_model(req: ReloadRequest):
    model_id = req.model_id or current_model_id
    await load_model(model_id)
    return {"status": "reloaded", "model_id": current_model_id}


@app.post("/shutdown")
async def shutdown(req: ShutdownRequest, background_tasks: BackgroundTasks):
    LOG.info("Shutdown requested: %s", req.reason)

    def _exit():
        os._exit(0)

    background_tasks.add_task(_exit)
    return {"status": "shutting down", "reason": req.reason}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": generator is not None,
        "mode": current_mode["name"],
        "queue_size": job_queue.qsize() if job_queue else None,
    }


if __name__ == "__main__":
    import uvicorn

    LOG.info("Run: uvicorn local_llm_server_reload_v2:app --host 0.0.0.0 --port 8000")
    uvicorn.run(
        "local_llm_server_reload_v2:app", host="0.0.0.0", port=8000, log_level="info"
    )
