import requests
from typing import Optional, Dict, Any
import time


class LocalLLMClient:
    """
    Python client for the Local LLM server with job queue.
    Supports enqueueing, polling results, mode switching, reload, shutdown, health.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """
        :param base_url: Base URL of the local LLM server
        :param timeout: Default HTTP timeout for requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---------------- Generate ----------------
    def enqueue(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Enqueue a new generation job, returns job_id"""
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
        return resp.json()["job_id"]

    def get_result(self, job_id: str) -> Dict[str, Any]:
        """Poll the server for job result or status"""
        resp = requests.get(f"{self.base_url}/result/{job_id}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def wait_for_result(
        self, job_id: str, poll_interval: float = 1.0, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Poll until job is complete or timeout occurs.
        :param poll_interval: seconds between polls
        :param timeout: total seconds to wait (None = infinite)
        """
        start_time = time.time()
        while True:
            r = self.get_result(job_id)
            if r["status"] in ("done", "error"):
                return r
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
            time.sleep(poll_interval)

    # ---------------- Mode ----------------
    def set_mode(
        self, mode: str, custom_system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Switch summarization mode"""
        resp = requests.post(
            f"{self.base_url}/mode",
            json={"mode": mode, "custom_system_prompt": custom_system_prompt},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ---------------- Reload model ----------------
    def reload_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Reload the model (HF model id), defaults to current"""
        resp = requests.post(
            f"{self.base_url}/reload", json={"model_id": model_id}, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    # ---------------- Shutdown ----------------
    def shutdown(self, reason: Optional[str] = "requested") -> Dict[str, Any]:
        """Shutdown server"""
        resp = requests.post(
            f"{self.base_url}/shutdown", json={"reason": reason}, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    # ---------------- Health ----------------
    def health(self) -> Dict[str, Any]:
        """Get server health"""
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
