"""
Local LLM Service - Embedded LLM using llama-cpp-python

Runs a lightweight model (e.g., Qwen2-0.5B, TinyLlama) directly in the worker container.
No external service required - model runs in-process.

Note: llama-cpp-python can crash with GGML_ASSERT in certain environments
(limited memory, incompatible CPU instructions, etc). This module uses
defensive programming to catch these issues and mark the service as unavailable.
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

# Thread pool for running blocking LLM inference (single worker to avoid threading issues)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local_llm")

# Lazy-loaded model instance
_llm_instance: Optional[Any] = None

# Flag to mark if model loading has failed permanently
_model_load_failed: bool = False
_model_load_error: Optional[str] = None


def _get_model():
    """
    Lazily load the LLM model on first use.

    Raises:
        RuntimeError: If model cannot be loaded or has previously failed
    """
    global _llm_instance, _model_load_failed, _model_load_error

    # If we've already tried and failed, don't retry
    if _model_load_failed:
        raise RuntimeError(f"Local LLM permanently unavailable: {_model_load_error}")

    if _llm_instance is not None:
        return _llm_instance

    try:
        from llama_cpp import Llama
    except ImportError as e:
        _model_load_failed = True
        _model_load_error = "llama-cpp-python not installed"
        logger.error(f"[LocalLLMService] {_model_load_error}. Run: pip install llama-cpp-python")
        raise RuntimeError(_model_load_error) from e

    model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "/app/models/qwen2-0_5b-instruct-q4_k_m.gguf")
    # Reduced defaults for Azure App Service compatibility
    n_ctx = int(os.getenv("LOCAL_LLM_CONTEXT_SIZE", "512"))  # Reduced from 2048
    n_threads = int(os.getenv("LOCAL_LLM_THREADS", "2"))  # Reduced from 4

    if not os.path.exists(model_path):
        _model_load_failed = True
        _model_load_error = f"Model file not found: {model_path}"
        logger.error(f"[LocalLLMService] {_model_load_error}")
        raise RuntimeError(_model_load_error)

    logger.info(f"[LocalLLMService] Loading model from {model_path} (ctx={n_ctx}, threads={n_threads})...")

    try:
        _llm_instance = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=64,  # Smaller batch size for stability
            verbose=False,
            use_mmap=True,  # Memory-mapped file for efficiency
            use_mlock=False,  # Don't lock memory (may fail in containers)
        )

        # Validate model works with a simple test generation
        logger.info("[LocalLLMService] Testing model with a simple prompt...")
        test_output = _llm_instance("Hello", max_tokens=1, temperature=0.0)
        if not test_output or "choices" not in test_output:
            raise RuntimeError("Model test generation returned invalid output")

        logger.info("[LocalLLMService] Model loaded and validated successfully")
        return _llm_instance

    except Exception as e:
        _model_load_failed = True
        _model_load_error = f"Model loading/validation failed: {type(e).__name__}: {e}"
        logger.error(f"[LocalLLMService] {_model_load_error}")
        _llm_instance = None
        raise RuntimeError(_model_load_error) from e


def is_model_available() -> bool:
    """Check if the model is available without loading it."""
    if _model_load_failed:
        return False
    model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "/app/models/qwen2-0_5b-instruct-q4_k_m.gguf")
    return os.path.exists(model_path)


class LocalLLMService:
    """
    Embedded local LLM service using llama-cpp-python.

    Runs a quantized model (GGUF format) directly in the worker process.
    Good for RAG tasks where speed is less critical than cost.

    IMPORTANT: This service uses conservative settings for Azure App Service:
    - Reduced context size (512 vs 2048) to avoid memory issues
    - Reduced threads (2 vs 4) to avoid CPU contention
    - Single worker thread pool to avoid threading issues
    - Model validation on first load to catch GGML_ASSERT early

    Recommended models (small, CPU-friendly):
    - qwen2-0.5b-instruct-q4_k_m.gguf (~400MB)
    - tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (~700MB)
    - phi-2.Q4_K_M.gguf (~1.6GB)
    """

    def __init__(self) -> None:
        self.model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "/app/models/qwen2-0_5b-instruct-q4_k_m.gguf")
        self.max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "256"))  # Reduced from 512
        self._available: Optional[bool] = None
        self._permanently_failed: bool = False

    def is_available(self) -> bool:
        """Check if the local LLM is available (without loading the model)."""
        if self._permanently_failed:
            return False

        if self._available is not None:
            return self._available

        # Check module-level failure flag first
        if _model_load_failed:
            logger.warning(f"[LocalLLMService] Model previously failed: {_model_load_error}")
            self._available = False
            self._permanently_failed = True
            return False

        # Just check if the model file exists - don't load it yet
        # Model will be loaded lazily on first actual use
        if os.path.exists(self.model_path):
            self._available = True
            logger.info(f"[LocalLLMService] Model file found: {self.model_path}")
        else:
            logger.warning(f"[LocalLLMService] Model file not found: {self.model_path}")
            self._available = False

        return self._available

    def mark_unavailable(self, reason: str) -> None:
        """Mark this service as permanently unavailable (called after runtime failures)."""
        self._available = False
        self._permanently_failed = True
        logger.warning(f"[LocalLLMService] Marked as permanently unavailable: {reason}")

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
            RuntimeError: If the model is unavailable or crashes
        """
        if self._permanently_failed:
            raise RuntimeError("Local LLM is permanently unavailable")

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
        except RuntimeError as e:
            # Model loading or validation failed - mark as permanently unavailable
            if "permanently unavailable" in str(e) or "Model loading" in str(e):
                self.mark_unavailable(str(e))
            raise
        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            logger.error(f"[LocalLLMService] Generation failed: {error_type}: {error_str}")

            # Check for GGML_ASSERT or other fatal errors
            if "GGML_ASSERT" in error_str or "Segmentation fault" in error_str:
                self.mark_unavailable(f"GGML crash: {error_str[:100]}")

            raise

    def _generate_sync(self, prompt: str, response_format: str) -> Dict[str, Any]:
        """Synchronous generation - runs in thread pool."""
        try:
            llm = _get_model()
        except RuntimeError:
            # Model loading failed - propagate to caller
            raise

        # Truncate prompt if too long to avoid context overflow
        # Use a conservative estimate: ~4 chars per token
        max_prompt_chars = 1500  # ~375 tokens, leaving room for output
        if len(prompt) > max_prompt_chars:
            logger.warning(f"[LocalLLMService] Truncating prompt from {len(prompt)} to {max_prompt_chars} chars")
            prompt = prompt[:max_prompt_chars] + "\n\n[Prompt truncated]"

        # For JSON output, add instruction to the prompt
        if response_format == "json":
            prompt = f"{prompt}\n\nRespond with valid JSON only, no explanation."

        logger.debug(
            f"[LocalLLMService] Generating response (max_tokens={self.max_tokens}, prompt_len={len(prompt)})..."
        )

        try:
            # Generate response with conservative settings
            output = llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=LLM_TEMPERATURE,
                stop=["</s>", "<|endoftext|>", "<|im_end|>"],
                echo=False,
            )
        except Exception as e:
            logger.error(f"[LocalLLMService] Generation call failed: {type(e).__name__}: {e}")
            raise

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
