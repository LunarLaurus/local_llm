from client.llm_client import LocalLLMClient
from typing import Optional, Dict, Any


class LLMWrapper:
    """
    High-level wrapper around LocalLLMClient to combine enqueue + wait.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ):
        """
        :param base_url: Local LLM server URL
        :param poll_interval: seconds between polling job results
        :param timeout: total seconds to wait for job completion (None = unlimited)
        """
        self.client = LocalLLMClient(base_url=base_url)
        self.poll_interval = poll_interval
        self.timeout = timeout

    # ---------------- Combined enqueue + wait ----------------
    def summarize_code(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Submit a code snippet (C, ASM, file, etc.), wait for the job, and return the final summary.
        """
        job_id = self.client.enqueue(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = self.client.wait_for_result(
            job_id, poll_interval=self.poll_interval, timeout=self.timeout
        )
        if result["status"] == "done":
            return result["result"]
        elif result["status"] == "cancelled":
            raise RuntimeError(f"Job {job_id} was cancelled")
        else:
            raise RuntimeError(f"Job failed: {result.get('error', 'unknown error')}")

    # ---------------- Mode management ----------------
    def set_mode(
        self, mode: str, custom_system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Switch summarization mode.
            c, asm, file, python, java, cpp, custom
        """
        return self.client.set_mode(mode, custom_system_prompt)

    def set_custom_prompt(self, system_prompt: str) -> Dict[str, Any]:
        """
        Convenience function to set the LLMWrapper to custom mode
        with a user-defined system prompt.

        :param llm: LLMWrapper instance
        :param system_prompt: the system prompt to use
        :return: server response dict
        """
        return self.client.set_mode("custom", custom_system_prompt=system_prompt)

    # ---------------- Model management ----------------
    def reload_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Reload HF model, defaults to current."""
        return self.client.reload_model(model_id)

    # ---------------- Health ----------------
    def health(self) -> Dict[str, Any]:
        """Check server health."""
        return self.client.health()

    # ---------------- Shutdown ----------------
    def shutdown(self, reason: Optional[str] = "requested") -> Dict[str, Any]:
        """Shutdown the server."""
        return self.client.shutdown(reason)

    # ---------------- Cancel ----------------
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return self.client.cancel(job_id)
