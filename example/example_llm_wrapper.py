# example_llm_wrapper.py
from llm_helpers import LLMWrapper


def main():
    # Initialize wrapper
    llm = LLMWrapper(base_url="http://localhost:8000", poll_interval=1.0, timeout=30.0)

    # Set summarization mode
    print("Setting mode to 'code_summary'...")
    resp = llm.set_mode("code_summary")
    print("Mode response:", resp)

    # Summarize some code
    code_snippet = """
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)
    """
    print("Submitting code for summarization...")
    summary = llm.summarize_code(
        user_prompt=code_snippet,
        system_prompt="Summarize this Python function in plain English.",
        max_tokens=50,
        temperature=0.5,
    )
    print("Summary result:", summary)

    # Check server health
    health = llm.health()
    print("Server health:", health)

    # Reload model (optional)
    reload_resp = llm.reload_model()
    print("Reload response:", reload_resp)

    # Shutdown server (optional)
    # shutdown_resp = llm.shutdown("example finished")
    # print("Shutdown response:", shutdown_resp)


if __name__ == "__main__":
    main()
