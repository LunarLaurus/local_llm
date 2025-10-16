from llm_helpers import LLMWrapper
import uuid


def main():
    # Initialize wrapper
    llm = LLMWrapper(base_url="http://localhost:8000", poll_interval=1.0, timeout=30.0)

    # ---------------- Normal mode flow ----------------
    print("Setting mode to 'python'...")
    resp = llm.set_mode("python")
    print("Mode response:", resp)

    code_snippet_python = """
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)
    """
    print("Submitting Python code for summarization...")
    summary_python = llm.summarize_code(
        user_prompt=code_snippet_python,
    )
    print("Python summary result:\n", summary_python)

    # ---------------- Health check ----------------
    health = llm.health()
    print("Server health:", health)

    # ---------------- Reload model ----------------
    reload_resp = llm.reload_model()
    print("Reload response:", reload_resp)

    # ---------------- Custom mode flow (niche language) ----------------
    custom_prompt = (
        "You are an expert in the R programming language. "
        "Summarize the following code clearly and concisely."
    )
    print("Setting custom mode for R code...")
    resp_custom = llm.set_custom_prompt(custom_prompt)
    print("Custom mode response:", resp_custom)

    code_snippet_r = """
    factorial <- function(n) {
        if (n == 0) {
            return(1)
        } else {
            return(n * factorial(n - 1))
        }
    }
    """
    print("Submitting R code for summarization...")
    summary_r = llm.summarize_code(
        user_prompt=code_snippet_r,
        max_tokens=50,
        temperature=0.5,
    )
    print("R summary result:\n", summary_r)

    # ---------------- Cancel a dummy job ----------------
    dummy_job_id = str(uuid.uuid4())
    print("Cancelling a dummy job:", dummy_job_id)
    cancel_resp = llm.cancel_job(dummy_job_id)
    print("Cancel response:", cancel_resp)

    # ---------------- Optional: shutdown ----------------
    # shutdown_resp = llm.shutdown("example finished")
    # print("Shutdown response:", shutdown_resp)


if __name__ == "__main__":
    main()
