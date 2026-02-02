"""
Hybrid LLM Service - Groq for critical tasks, Local LLM for others

Cost-conscious strategy with priority-based routing:
- HIGH priority tasks: Use Groq (fast, paid) - limited to MAX_GROQ_CALLS_PER_REQUEST
- LOW priority tasks: Use Local LLM directly (free, in-container)
- Fallback: If Groq fails, always fall back to Local LLM
"""

import contextvars
import os
from enum import Enum
from typing import Any, Dict, Optional

from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService, RateLimitError
from app.services.llms.local_llm_service import LocalLLMService
from app.services.llms.ollama_service import OllamaService

logger = get_logger(__name__)


class LLMPriority(Enum):
    """Priority levels for LLM calls."""

    HIGH = "high"  # Use Groq (critical tasks: query reformulation, initial fact extraction)
    LOW = "low"  # Use Local LLM (entity extraction, relation extraction, etc.)


# Context variable to track Groq calls per request/job
_groq_call_count: contextvars.ContextVar[int] = contextvars.ContextVar("groq_call_count", default=0)

# Maximum Groq calls allowed per request (for HIGH priority tasks)
MAX_GROQ_CALLS_PER_REQUEST = 5


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
    Hybrid LLM service with priority-based routing.

    Strategy:
    - HIGH priority: Use Groq (limited to 5 calls per request)
    - LOW priority: Use Local LLM directly (unlimited, free)
    - On Groq failure: Always fall back to Local LLM

    Call reset_groq_counter() at the start of each new job/request.
    """

    groq_service: Optional[GroqService]
    local_llm_service: Optional[LocalLLMService]
    ollama_service: Optional[OllamaService]

    def __init__(self) -> None:
        # Primary: Groq (fast, paid) - for HIGH priority tasks
        try:
            self.groq_service = GroqService()
            self.groq_available = True
            logger.info("[HybridLLMService] Groq service available (for high-priority tasks)")
        except Exception as e:
            logger.warning(f"[HybridLLMService] Groq service unavailable: {e}")
            self.groq_service = None
            self.groq_available = False

        # Fallback 1: Local in-container LLM (free) - for LOW priority & fallback
        # Can be disabled with LOCAL_LLM_ENABLED=false if model is crashing
        local_llm_enabled = os.getenv("LOCAL_LLM_ENABLED", "true").lower() == "true"
        if not local_llm_enabled:
            logger.info("[HybridLLMService] Local LLM disabled via LOCAL_LLM_ENABLED=false")
            self.local_llm_service = None
            self.local_llm_available = False
        else:
            try:
                self.local_llm_service = LocalLLMService()
                self.local_llm_available = self.local_llm_service.is_available()
                if self.local_llm_available:
                    logger.info("[HybridLLMService] Local LLM available (for low-priority & fallback)")
                else:
                    logger.warning("[HybridLLMService] Local LLM model not found (will use Groq for all tasks)")
            except Exception as e:
                logger.warning(f"[HybridLLMService] Local LLM service unavailable: {e}")
                self.local_llm_service = None
                self.local_llm_available = False

        # Fallback 2: Ollama (external service) - disabled by default in Azure
        # Only enable if OLLAMA_HOST is explicitly set to something other than 'ollama'
        ollama_host = os.getenv("OLLAMA_HOST", "ollama")
        if ollama_host == "ollama":
            # Default value means Ollama is not configured - skip it
            logger.info("[HybridLLMService] Ollama not configured (OLLAMA_HOST=ollama), skipping")
            self.ollama_service = None
            self.ollama_available = False
        else:
            try:
                self.ollama_service = OllamaService()
                self.ollama_available = True
                logger.info(f"[HybridLLMService] Ollama configured at {ollama_host}")
            except Exception as e:
                logger.warning(f"[HybridLLMService] Ollama service unavailable: {e}")
                self.ollama_service = None
                self.ollama_available = False

        if not self.groq_available and not self.local_llm_available and not self.ollama_available:
            raise RuntimeError("No LLM services available (Groq, LocalLLM, or Ollama)")

    async def ainvoke(
        self,
        prompt: str,
        response_format: str = "text",
        priority: LLMPriority = LLMPriority.HIGH,
    ) -> Dict[str, Any]:
        """
        Calls LLM with priority-based routing and automatic fallback.

        Args:
            prompt: The prompt to send to the LLM
            response_format: "text" or "json"
            priority: HIGH = use Groq (if quota available), LOW = use Local LLM directly

        Strategy:
        - LOW priority: Always use Local LLM (free, no API costs)
        - HIGH priority: Use Groq if quota available, else Local LLM
        - On any Groq failure: Fall back to Local LLM
        """
        # LOW priority tasks always use Local LLM
        if priority == LLMPriority.LOW:
            logger.debug("[HybridLLMService] LOW priority task -> using Local LLM")
            return await self._call_free_llm(prompt, response_format)

        # HIGH priority tasks use Groq (if available and quota not exhausted)
        current_groq_calls = get_groq_call_count()

        if current_groq_calls >= MAX_GROQ_CALLS_PER_REQUEST:
            logger.info(
                f"[HybridLLMService] Groq quota exhausted "
                f"({current_groq_calls}/{MAX_GROQ_CALLS_PER_REQUEST}), using Local LLM"
            )
            return await self._call_free_llm(prompt, response_format)

        # Try Groq for HIGH priority (with fallback on failure)
        if self.groq_available and self.groq_service:
            try:
                call_num = _increment_groq_counter()
                logger.info(
                    f"[HybridLLMService] HIGH priority -> Groq " f"(call {call_num}/{MAX_GROQ_CALLS_PER_REQUEST})"
                )
                result = await self.groq_service.ainvoke(prompt, response_format, max_retries=2)
                logger.debug(f"[HybridLLMService] Groq succeeded (call {call_num})")
                return result
            except RateLimitError as e:
                logger.warning(f"[HybridLLMService] Groq rate limited: {e}. Falling back to Local LLM...")
            except Exception as e:
                logger.warning(f"[HybridLLMService] Groq failed: {e}. Falling back to Local LLM...")

        # Fallback to Local LLM on any Groq failure
        return await self._call_free_llm(prompt, response_format)

    async def _call_free_llm(self, prompt: str, response_format: str) -> Dict[str, Any]:
        """
        Call free LLM service with Groq as final fallback.
        Priority: Local LLM (in-container) > Ollama (external) > Groq (paid, last resort)
        """
        # Try local in-container LLM first
        if self.local_llm_available and self.local_llm_service:
            try:
                logger.debug("[HybridLLMService] Using Local LLM (in-container, free)...")
                result = await self.local_llm_service.ainvoke(prompt, response_format)
                logger.debug("[HybridLLMService] Local LLM succeeded")
                return result
            except Exception as e:
                error_str = str(e)
                logger.warning(f"[HybridLLMService] Local LLM failed: {e}, trying Ollama...")

                # If Local LLM is permanently unavailable, mark it so
                if "permanently unavailable" in error_str or hasattr(self.local_llm_service, "_permanently_failed"):
                    if getattr(self.local_llm_service, "_permanently_failed", False):
                        self.local_llm_available = False
                        logger.warning("[HybridLLMService] Local LLM marked as permanently unavailable")

        # Fallback to Ollama
        if self.ollama_available and self.ollama_service:
            try:
                logger.info("[HybridLLMService] Using Ollama (external, free)...")
                result = await self.ollama_service.ainvoke(prompt, response_format)
                logger.info("[HybridLLMService] Ollama succeeded")
                return result
            except Exception as e:
                logger.warning(f"[HybridLLMService] Ollama failed: {e}")

        # Final fallback: Use Groq even for LOW priority if no free options work
        # This ensures the pipeline completes rather than failing
        if self.groq_available and self.groq_service:
            try:
                call_num = _increment_groq_counter()
                logger.warning(
                    f"[HybridLLMService] No free LLM available, using Groq as last resort "
                    f"(call {call_num}/{MAX_GROQ_CALLS_PER_REQUEST})"
                )
                result = await self.groq_service.ainvoke(prompt, response_format, max_retries=2)
                logger.info("[HybridLLMService] Groq fallback succeeded")
                return result
            except Exception as e:
                logger.error(f"[HybridLLMService] Groq fallback also failed: {e}")
                raise

        raise RuntimeError("No LLM service available (Local LLM, Ollama, or Groq)")
