from setuptools import setup, find_packages

setup(
    name="laurus-local-llm",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "transformers",
        "torch",
        "bitsandbytes",
        "accelerate",
        "pyyaml",
        "requests",
        "pydantic",
    ],
    python_requires=">=3.10",
)
