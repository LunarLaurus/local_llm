import threading
from typing import Literal, Optional
from .config import DEFAULT_MODEL_ID, DEFAULT_MAX_TOKENS, DEFAULT_TEMP, MODES
from .lauruslog import LOG


class Generator:
    """
    Wraps a HuggingFace LLM pipeline with configurable bitness, max tokens, and temperature.
    Supports singleton instance for global access. Automatically falls back to CPU if no GPU.
    """

    _instance: Optional["Generator"] = None
    current_mode = {"name": "c", "system_prompt": MODES["c"]}

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMP,
        bitness: Literal["4bit", "8bit", "16bit"] = "16bit",
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.bitness = bitness
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._thread_lock = threading.Lock()

    def load_model(self, model_id: Optional[str] = None):
        """Load HuggingFace model and tokenizer with GPU fallback to CPU."""
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            pipeline,
            BitsAndBytesConfig,
        )
        import torch

        model_id = model_id or self.model_id
        LOG.info("Loading model %s with %s", model_id, self.bitness)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # Detect if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            LOG.info("No GPU detected. Optimizing model for CPU inference.")
        else:
            LOG.info("GPU detected. Using CUDA for inference.")

        try:
            if self.bitness == "4bit" and device != "cpu":
                # Only load 4-bit quantized if GPU exists
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                )
                LOG.info("Loaded 4-bit quantized model")
            elif self.bitness == "8bit" and device != "cpu":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                )
                LOG.info("Loaded 8-bit quantized model")
            else:
                # CPU fallback or full precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=None if device == "cpu" else "auto",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    trust_remote_code=True,
                )
                LOG.info("Loaded full precision model on %s", device)
        except Exception as e:
            LOG.warning(
                "Failed to load model with %s, falling back to CPU float32: %s",
                self.bitness,
                e,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=None,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        if device == "cpu":
            # Optimize CPU performance
            import torch

            torch.set_num_threads(max(1, torch.get_num_threads()))
            LOG.info("CPU threads available: %d", torch.get_num_threads())

        self.model_id = model_id
        LOG.info("Model pipeline ready on %s", device)

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Thread-safe text generation."""
        full_prompt = f"{system_prompt or ''}\n\nUser:\n{user_prompt}\n\nAssistant:"
        max_new_tokens = max(1, min(2000, int(max_tokens or self.max_tokens)))
        temp = float(temperature if temperature is not None else self.temperature)

        assert self.tokenizer is not None, "Model tokenizer not loaded yet"
        assert self.pipeline is not None, "Model pipeline not loaded yet"
        with self._thread_lock:
            outputs = self.pipeline(
                full_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=(temp > 0),
                temperature=temp,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = outputs[0].get("generated_text", "")
        if "Assistant:" in text:
            return text.split("Assistant:")[-1].strip()
        return text.replace(full_prompt, "").strip()

    @classmethod
    def get_instance(cls) -> "Generator":
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, instance: "Generator"):
        """Set the singleton instance explicitly."""
        cls._instance = instance
