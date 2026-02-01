"""
Hybrid LLM Service - Groq with Local LLM fallback

Implements a cost-conscious strategy:
- Groq (fast, paid): Limited to MAX_GROQ_CALLS_PER_REQUEST per job/request
- Local LLM (free, in-container): Used for remaining calls after Groq quota exhausted
"""

import contextvars
from typing import Any, Dict, Optional

from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService, RateLimitError
from app.services.llms.local_llm_service import LocalLLMService
from app.services.llms.ollama_service import OllamaService

logger = get_logger(__name__)

# Context variable to track Groq calls per request/job
_groq_call_count: contextvars.ContextVar[int] = contextvars.ContextVar("groq_call_count", default=0)

# Maximum Groq calls allowed per request before falling back to local LLM
MAX_GROQ_CALLS_PER_REQUEST = 3


def reset_groq_counter() -> None:
    """Reset Groq call counter for a new request/job."""
    _groq_call_count.set(0)
    logger.debug("[HybridLLMService] Groq call counter reset")


def get_groq_call_count() -> int:
    """Get current Groq call count for this request."""
    return _groq_call_count.get()


def _increment_groq_counter() -> int:
    """Increment and return the new Groq call count."""
    current = _groq_call_count.get()
    new_count = current + 1
    _groq_call_count.set(new_count)
    return new_count


class HybridLLMService:
    """
    Hybrid LLM service that uses Groq (fast) with Local LLM (free) fallback.

    Cost-saving strategy:
    - First N calls (MAX_GROQ_CALLS_PER_REQUEST) use Groq for speed
    - Subsequent calls use local in-container LLM (free)
    - Falls back on Groq errors (rate limit, API errors)

    Fallback priority: LocalLLMService (in-container) > OllamaService (external)

    Call reset_groq_counter() at the start of each new job/request.
    """

    groq_service: Optional[GroqService]
    local_llm_service: Optional[LocalLLMService]
    ollama_service: Optional[OllamaService]

    def __init__(self) -> None:
        # Primary: Groq (fast, paid)
        try:
            self.groq_service = GroqService()
            self.groq_available = True
        except Exception as e:
            logger.warning(f"[HybridLLMService] Groq service unavailable: {e}")
            self.groq_service = None
            self.groq_available = False

        # Fallback 1: Local in-container LLM (free, no network)
        try:
            self.local_llm_service = LocalLLMService()
            self.local_llm_available = self.local_llm_service.is_available()
            if self.local_llm_available:
                logger.info("[HybridLLMService] Local LLM available (in-container)")
        except Exception as e:
            logger.warning(f"[HybridLLMService] Local LLM service unavailable: {e}")
            self.local_llm_service = None
            self.local_llm_available = False

        # Fallback 2: Ollama (external service)
        try:
            self.ollama_service = OllamaService()
            self.ollama_available = True
        except Exception as e:
            logger.warning(f"[HybridLLMService] Ollama service unavailable: {e}")
            self.ollama_service = None
            self.ollama_available = False

        if not self.groq_available and not self.local_llm_available and not self.ollama_available:
            raise RuntimeError("No LLM services available (Groq, LocalLLM, or Ollama)")

    async def ainvoke(self, prompt: str, response_format: str = "text", force_local: bool = False) -> Dict[str, Any]:
        """
        Calls LLM with automatic fallback and cost management.

        Args:
            prompt: The prompt to send to the LLM
            response_format: "text" or "json"
            force_local: If True, skip Groq and use local LLM directly

        Strategy:
        1. If Groq quota not exhausted (< MAX_GROQ_CALLS_PER_REQUEST), try Groq
        2. If Groq fails or quota exhausted, use Local LLM (in-container)
        3. If Local LLM unavailable, try Ollama (external service)
        """
        current_groq_calls = get_groq_call_count()

        # Check if we should skip Groq due to quota
        if force_local or current_groq_calls >= MAX_GROQ_CALLS_PER_REQUEST:
            if current_groq_calls >= MAX_GROQ_CALLS_PER_REQUEST:
                logger.info(
                    f"[HybridLLMService] Groq quota exhausted "
                    f"({current_groq_calls}/{MAX_GROQ_CALLS_PER_REQUEST}), using free LLM"
                )
            return await self._call_free_llm(prompt, response_format)

        # Try Groq (with retries on rate limit)
        if self.groq_available and self.groq_service:
            for attempt in range(3):
                try:
                    call_num = _increment_groq_counter()
                    logger.debug(
                        f"[HybridLLMService] Groq call {call_num}/{MAX_GROQ_CALLS_PER_REQUEST} "
                        f"(attempt {attempt + 1}/3)"
                    )
                    result = await self.groq_service.ainvoke(prompt, response_format, max_retries=1)
                    logger.debug(f"[HybridLLMService] Groq succeeded (call {call_num})")
                    return result
                except RateLimitError as e:
                    if attempt < 2:
                        logger.warning(
                            f"[HybridLLMService] Groq rate limited (attempt {attempt + 1}/3): {e}. Retrying..."
                        )
                        continue
                    else:
                        logger.warning("[HybridLLMService] Groq rate limited after 3 attempts. Using free LLM...")
                        break
                except Exception as e:
                    logger.warning(f"[HybridLLMService] Groq failed (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        continue
                    else:
                        logger.warning("[HybridLLMService] Groq failed after 3 attempts. Using free LLM...")
                        break

        # Fallback to free LLM
        return await self._call_free_llm(prompt, response_format)

    async def _call_free_llm(self, prompt: str, response_format: str) -> Dict[str, Any]:
        """
        Call free LLM service.
        Priority: Local LLM (in-container) > Ollama (external)
        """
        # Try local in-container LLM first
        if self.local_llm_available and self.local_llm_service:
            try:
                logger.info("[HybridLLMService] Using Local LLM (in-container, free)...")
                result = await self.local_llm_service.ainvoke(prompt, response_format)
                logger.info("[HybridLLMService] Local LLM succeeded")
                return result
            except Exception as e:
                logger.warning(f"[HybridLLMService] Local LLM failed: {e}, trying Ollama...")

        # Fallback to Ollama
        if self.ollama_available and self.ollama_service:
            try:
                logger.info("[HybridLLMService] Using Ollama (external, free)...")
                result = await self.ollama_service.ainvoke(prompt, response_format)
                logger.info("[HybridLLMService] Ollama succeeded")
                return result
            except Exception as e:
                logger.error(f"[HybridLLMService] Ollama failed: {e}")
                raise

        raise RuntimeError("No free LLM service available (Local LLM or Ollama)")
