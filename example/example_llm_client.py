# example_llm_client.py
from llm_client import LocalLLMClient


def main():
    # Initialize client
    client = LocalLLMClient(base_url="http://localhost:8000", timeout=30.0)

    # Enqueue a job
    code_snippet = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """
    job_id = client.enqueue(
        user_prompt=code_snippet,
        system_prompt="Explain this Python function in simple terms.",
        max_tokens=50,
        temperature=0.5,
    )
    print("Job submitted, job_id:", job_id)

    # Wait for result
    result = client.wait_for_result(job_id, poll_interval=1.0, timeout=30.0)
    if result["status"] == "done":
        print("Job completed! Result:", result["result"])
    else:
        print("Job failed:", result.get("error"))

    # Change mode
    mode_resp = client.set_mode("code_summary")
    print("Mode set response:", mode_resp)

    # Reload model
    reload_resp = client.reload_model()
    print("Reload response:", reload_resp)

    # Check health
    health_resp = client.health()
    print("Server health:", health_resp)

    # Shutdown server (optional)
    # shutdown_resp = client.shutdown("Finished example")
    # print("Shutdown response:", shutdown_resp)


if __name__ == "__main__":
    main()
