import requests
from typing import Optional, Dict, Any
import time

from laurus_llm.lauruslog import LOG


class LocalLLMClient:
    """
    Python client for the Local LLM server with job queue.
    Supports enqueueing, polling results, mode switching, reload, shutdown, health.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        LOG.info(
            "Initialized LocalLLMClient with base_url=%s, timeout=%.1fs",
            self.base_url,
            self.timeout,
        )

    # ---------------- Generate ----------------
    def enqueue(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Enqueue a new generation job, returns job_id"""
        LOG.info(
            "Enqueueing job (max_tokens=%s, temperature=%s)", max_tokens, temperature
        )
        try:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            LOG.info("Job enqueued successfully, job_id=%s", job_id)
            return job_id
        except Exception as e:
            LOG.exception("Failed to enqueue job")
            raise

    def get_result(self, job_id: str) -> Dict[str, Any]:
        """Poll the server for job result or status"""
        LOG.info("Fetching result for job_id=%s", job_id)
        try:
            resp = requests.get(
                f"{self.base_url}/result/{job_id}", timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            LOG.exception("Failed to get result for job_id=%s", job_id)
            raise

    def wait_for_result(
        self, job_id: str, poll_interval: float = 1.0, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        LOG.info(
            "Waiting for result of job_id=%s with poll_interval=%.1fs, timeout=%s",
            job_id,
            poll_interval,
            timeout,
        )
        while True:
            r = self.get_result(job_id)
            if r["status"] in ("done", "error", "cancelled"):
                LOG.info("Job %s completed with status=%s", job_id, r["status"])
                return r
            if timeout is not None and (time.time() - start_time) > timeout:
                LOG.warning("Job %s did not complete within %ds", job_id, timeout)
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
            time.sleep(poll_interval)

    # ---------------- Mode ----------------
    def set_mode(
        self, mode: str, custom_system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        LOG.info("Switching mode to '%s'", mode)
        try:
            resp = requests.post(
                f"{self.base_url}/mode",
                json={"mode": mode, "custom_system_prompt": custom_system_prompt},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            LOG.info("Mode switched to '%s'", mode)
            return resp.json()
        except Exception as e:
            LOG.exception("Failed to set mode '%s'", mode)
            raise

    # ---------------- Reload model ----------------
    def reload_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        LOG.info("Reloading model: %s", model_id or "(current)")
        try:
            resp = requests.post(
                f"{self.base_url}/reload",
                json={"model_id": model_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            LOG.info("Model reloaded successfully")
            return resp.json()
        except Exception as e:
            LOG.exception("Failed to reload model")
            raise

    # ---------------- Shutdown ----------------
    def shutdown(self, reason: Optional[str] = "requested") -> Dict[str, Any]:
        LOG.info("Sending shutdown request (reason=%s)", reason)
        try:
            resp = requests.post(
                f"{self.base_url}/shutdown",
                json={"reason": reason},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            LOG.info("Shutdown request successful")
            return resp.json()
        except Exception as e:
            LOG.exception("Failed to shutdown server")
            raise

    # ---------------- Cancel ----------------
    def cancel(self, job_id: str) -> Dict[str, Any]:
        LOG.info("Cancelling job_id=%s", job_id)
        try:
            resp = requests.post(
                f"{self.base_url}/cancel/{job_id}", timeout=self.timeout
            )
            resp.raise_for_status()
            LOG.info("Job %s cancelled successfully", job_id)
            return resp.json()
        except Exception as e:
            LOG.exception("Failed to cancel job_id=%s", job_id)
            raise

    # ---------------- Health ----------------
    def health(self) -> Dict[str, Any]:
        LOG.debug("Checking server health")
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            resp.raise_for_status()
            LOG.info("Server health OK")
            return resp.json()
        except Exception as e:
            LOG.exception("Failed to get server health")
            raise
