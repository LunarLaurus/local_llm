# client_examples.py
"""
Simple client wrappers for local LLM server
"""
import requests
from typing import Optional

BASE_URL = "http://localhost:8000"


def enqueue(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    resp = requests.post(
        f"{BASE_URL}/generate",
        json={
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def get_result(job_id: str) -> dict:
    resp = requests.get(f"{BASE_URL}/result/{job_id}")
    resp.raise_for_status()
    return resp.json()


def set_mode(mode: str, custom_system_prompt: Optional[str] = None) -> dict:
    resp = requests.post(
        f"{BASE_URL}/mode",
        json={"mode": mode, "custom_system_prompt": custom_system_prompt},
    )
    resp.raise_for_status()
    return resp.json()


def reload_model(model_id: Optional[str] = None) -> dict:
    resp = requests.post(f"{BASE_URL}/reload", json={"model_id": model_id})
    resp.raise_for_status()
    return resp.json()


def shutdown(reason: Optional[str] = "requested") -> dict:
    resp = requests.post(f"{BASE_URL}/shutdown", json={"reason": reason})
    resp.raise_for_status()
    return resp.json()


def health() -> dict:
    resp = requests.get(f"{BASE_URL}/health")
    resp.raise_for_status()
    return resp.json()


# Example usage
if __name__ == "__main__":
    job_id = enqueue("Summarize this C function:\n\nint add(int a,int b){return a+b;}")
    print("Enqueued job:", job_id)

    import time

    while True:
        r = get_result(job_id)
        print("Job status:", r["status"])
        if r["status"] == "done" or r["status"] == "error":
            print(r)
            break
        time.sleep(1)
