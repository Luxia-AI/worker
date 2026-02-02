"""
Local LLM Service - Embedded LLM using llama-cpp-python

Runs a lightweight model (e.g., Qwen2-0.5B, TinyLlama) directly in the worker container.
No external service required - model runs in-process.
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from app.constants.config import LLM_TEMPERATURE
from app.core.logger import get_logger

logger = get_logger(__name__)

# Timeout for local LLM generation (seconds)
# Reduced to 10s to fail fast - if model is slow/crashing, fallback to Groq quickly
LOCAL_LLM_TIMEOUT = int(os.getenv("LOCAL_LLM_TIMEOUT", "10"))

# Thread pool for running blocking LLM inference
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="local_llm")

# Lazy-loaded model instance
_llm_instance: Optional[Any] = None


def _get_model():
    """Lazily load the LLM model on first use."""
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("[LocalLLMService] llama-cpp-python not installed. Run: pip install llama-cpp-python")
        raise RuntimeError("llama-cpp-python not installed")

    model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "/app/models/qwen2-0_5b-instruct-q4_k_m.gguf")
    n_ctx = int(os.getenv("LOCAL_LLM_CONTEXT_SIZE", "2048"))
    n_threads = int(os.getenv("LOCAL_LLM_THREADS", "4"))

    if not os.path.exists(model_path):
        logger.error(f"[LocalLLMService] Model not found at {model_path}")
        raise RuntimeError(f"Model file not found: {model_path}")

    logger.info(f"[LocalLLMService] Loading model from {model_path}...")
    _llm_instance = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )
    logger.info("[LocalLLMService] Model loaded successfully")

    return _llm_instance


class LocalLLMService:
    """
    Embedded local LLM service using llama-cpp-python.

    Runs a quantized model (GGUF format) directly in the worker process.
    Good for RAG tasks where speed is less critical than cost.

    Recommended models (small, CPU-friendly):
    - qwen2-0.5b-instruct-q4_k_m.gguf (~400MB)
    - tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (~700MB)
    - phi-2.Q4_K_M.gguf (~1.6GB)
    """

    def __init__(self) -> None:
        self.model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "/app/models/qwen2-0_5b-instruct-q4_k_m.gguf")
        self.max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "512"))
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if the local LLM is available (without loading the model)."""
        if self._available is not None:
            return self._available

        # Just check if the model file exists - don't load it yet
        # Model will be loaded lazily on first actual use
        if os.path.exists(self.model_path):
            self._available = True
            logger.info(f"[LocalLLMService] Model file found: {self.model_path}")
        else:
            logger.warning(f"[LocalLLMService] Model file not found: {self.model_path}")
            self._available = False

        return self._available

    async def ainvoke(self, prompt: str, response_format: str = "text") -> Dict[str, Any]:
        """
        Generate text using the local LLM with timeout protection.

        Args:
            prompt: The prompt to send to the model
            response_format: "text" or "json"

        Returns:
            Dict with generated text or parsed JSON

        Raises:
            asyncio.TimeoutError: If generation takes longer than LOCAL_LLM_TIMEOUT
        """
        try:
            # Run blocking LLM inference in thread pool with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(_executor, self._generate_sync, prompt, response_format),
                timeout=LOCAL_LLM_TIMEOUT,
            )
            return result

        except asyncio.TimeoutError:
            logger.warning(f"[LocalLLMService] Generation timed out after {LOCAL_LLM_TIMEOUT}s")
            raise
        except Exception as e:
            logger.error(f"[LocalLLMService] Generation failed: {e}")
            raise

    def _generate_sync(self, prompt: str, response_format: str) -> Dict[str, Any]:
        """Synchronous generation - runs in thread pool."""
        llm = _get_model()

        # For JSON output, add instruction to the prompt
        if response_format == "json":
            prompt = f"{prompt}\n\nRespond with valid JSON only, no explanation."

        logger.debug(f"[LocalLLMService] Generating response (max_tokens={self.max_tokens})...")

        # Generate response
        output = llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=LLM_TEMPERATURE,
            stop=["</s>", "<|endoftext|>", "<|im_end|>"],
            echo=False,
        )

        text = output["choices"][0]["text"].strip()
        logger.debug(f"[LocalLLMService] Generated {len(text)} chars")

        # Parse JSON if requested
        if response_format == "json":
            if text:
                try:
                    # Try to extract JSON from the response
                    # Sometimes models add extra text before/after JSON
                    json_start = text.find("{")
                    json_end = text.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = text[json_start:json_end]
                        return json.loads(json_str)
                    # Try array format - if it's a list with one dict, unwrap it
                    json_start = text.find("[")
                    json_end = text.rfind("]") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = text[json_start:json_end]
                        parsed = json.loads(json_str)
                        # If it's a list with a single dict, return the dict
                        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                            return parsed[0]
                        return parsed
                    # Try parsing as-is
                    return json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"[LocalLLMService] Failed to parse JSON: {text[:200]}")
                    # Return error indicator instead of empty dict so callers know parsing failed
                    return {"_llm_error": "json_parse_failed", "raw_text": text[:500]}
            # Empty text
            return {"_llm_error": "empty_response"}

        return {"text": text}
