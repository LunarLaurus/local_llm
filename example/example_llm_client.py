from llm_client import LocalLLMClient


def main():
    # Initialize client
    client = LocalLLMClient(base_url="http://localhost:8000", timeout=30.0)

    # ---------------- Enqueue a job ----------------
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

    # ---------------- Wait for result ----------------
    try:
        result = client.wait_for_result(job_id, poll_interval=1.0, timeout=30.0)
        if result["status"] == "done":
            print("Job completed! Result:\n", result["result"])
        elif result["status"] == "cancelled":
            print("Job was cancelled.")
        else:
            print("Job failed:", result.get("error"))
    except TimeoutError as e:
        print("Timeout:", e)
        # Example: cancel the job if it timed out
        cancel_resp = client.cancel(job_id)
        print("Cancelled job response:", cancel_resp)

    # ---------------- Set custom mode ----------------
    custom_prompt = (
        "You are a Python code expert. Explain the function clearly and concisely."
    )
    mode_resp = client.set_mode("custom", custom_system_prompt=custom_prompt)
    print("Custom mode set response:", mode_resp)

    # ---------------- Reload model ----------------
    reload_resp = client.reload_model()
    print("Reload response:", reload_resp)

    # ---------------- Check health ----------------
    health_resp = client.health()
    print("Server health:", health_resp)

    # ---------------- Optional: Shutdown ----------------
    # shutdown_resp = client.shutdown("Finished example")
    # print("Shutdown response:", shutdown_resp)


if __name__ == "__main__":
    main()
