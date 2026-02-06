"""
Groq-only LLM Service

All LLM calls are routed to Groq. Local/ollama services are intentionally disabled.
"""

import contextvars
from enum import Enum
from typing import Any, Dict, Optional

from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService, RateLimitError

logger = get_logger(__name__)


class LLMPriority(Enum):
    """Priority levels for LLM calls."""

    HIGH = "high"  # Use Groq (critical tasks: query reformulation, initial fact extraction)
    LOW = "low"  # Use Local LLM (entity extraction, relation extraction, etc.)


# Context variable to track Groq calls per request/job
_groq_call_count: contextvars.ContextVar[int] = contextvars.ContextVar("groq_call_count", default=0)

# Maximum Groq calls allowed per request (informational only)
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
    Groq-only LLM service.
    Call reset_groq_counter() at the start of each new job/request (for metrics).
    """

    groq_service: Optional[GroqService]

    def __init__(self) -> None:
        # Primary and only: Groq
        try:
            self.groq_service = GroqService()
            self.groq_available = True
            logger.info("[HybridLLMService] Groq service available (Groq-only mode)")
        except Exception as e:
            logger.warning(f"[HybridLLMService] Groq service unavailable: {e}")
            self.groq_service = None
            self.groq_available = False

        if not self.groq_available:
            raise RuntimeError("Groq LLM service unavailable")

    async def ainvoke(
        self,
        prompt: str,
        response_format: str = "text",
        priority: LLMPriority = LLMPriority.HIGH,
        temperature: float | None = None,
    ) -> Dict[str, Any]:
        """
        Calls LLM with priority-based routing and automatic fallback.

        Args:
            prompt: The prompt to send to the LLM
            response_format: "text" or "json"
            priority: HIGH = use Groq (if quota available), LOW = use Local LLM directly
            temperature: Override default temperature (0.0-1.0, lower = more deterministic)

        Strategy:
        - All priorities use Groq.
        """
        # Try Groq for all priorities
        if self.groq_available and self.groq_service:
            try:
                call_num = _increment_groq_counter()
                logger.info(
                    f"[HybridLLMService] HIGH priority -> Groq " f"(call {call_num}/{MAX_GROQ_CALLS_PER_REQUEST})"
                )
                result = await self.groq_service.ainvoke(
                    prompt, response_format, max_retries=2, temperature=temperature
                )
                logger.debug(f"[HybridLLMService] Groq succeeded (call {call_num})")
                return result
            except RateLimitError as e:
                logger.warning(f"[HybridLLMService] Groq rate limited: {e}")
            except Exception as e:
                logger.warning(f"[HybridLLMService] Groq failed: {e}")

        raise RuntimeError("Groq LLM unavailable or request failed")
