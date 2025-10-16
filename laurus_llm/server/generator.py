import logging
import threading
from typing import Literal, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from config import DEFAULT_MODEL_ID, DEFAULT_MAX_TOKENS, DEFAULT_TEMP, MODES

LOG = logging.getLogger("laurus-llm")


class Generator:
    """
    Wraps a HuggingFace LLM pipeline with configurable bitness, max tokens, and temperature.
    """

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
        self.load_model(model_id)

    def load_model(self, model_id: Optional[str] = None):
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            pipeline,
            BitsAndBytesConfig,
        )

        model_id = model_id or self.model_id
        LOG.info("Loading model %s with %s", model_id, self.bitness)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        try:
            if self.bitness == "4bit":
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                )
                LOG.info("Loaded 4-bit quantized model")
            elif self.bitness == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                )
                LOG.info("Loaded 8-bit quantized model")
            else:
                # 16bit / default
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
                LOG.info("Loaded full precision model")
        except Exception as e:
            LOG.warning(
                "Failed to load model with %s, falling back to float16: %s",
                self.bitness,
                e,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True
            )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            device_map="auto",
        )
        self.model_id = model_id
        LOG.info("Model pipeline ready")

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        full_prompt = f"{system_prompt or ''}\n\nUser:\n{user_prompt}\n\nAssistant:"
        max_new_tokens = max(1, min(2000, int(max_tokens or self.max_tokens)))
        temp = float(temperature if temperature is not None else self.temperature)

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
