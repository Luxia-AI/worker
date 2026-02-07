"""Hybrid LLM service with per-job Groq quota hardening and degradation support."""

from __future__ import annotations

import asyncio
import contextvars
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService, RateLimitError

logger = get_logger(__name__)


class LLMPriority(Enum):
    """Priority levels for LLM calls."""

    HIGH = "high"  # Use Groq (critical tasks: query reformulation, initial fact extraction)
    LOW = "low"  # Use Local LLM (entity extraction, relation extraction, etc.)


@dataclass
class GroqJobState:
    job_id: str
    max_calls: int
    calls_used: int = 0
    burst_enabled: bool = False
    degraded_mode: bool = False
    skipped_calls: list[str] = field(default_factory=list)
    fallback_to_local: bool = False


_groq_job_state: contextvars.ContextVar[GroqJobState] = contextvars.ContextVar(
    "groq_job_state",
    default=GroqJobState(job_id="default", max_calls=5),
)

_DEGRADE_ORDER = {
    "relation_extraction": 1,
    "entity_extraction": 2,
    "fact_extraction": 3,
}


def _env_allow_groq_burst() -> bool:
    return (os.getenv("ALLOW_GROQ_BURST", "false") or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_max_calls() -> int:
    try:
        return max(1, int(os.getenv("GROQ_MAX_CALLS_PER_JOB", "5")))
    except Exception:
        return 5


def reset_groq_counter(job_id: str | None = None, max_calls: int | None = None) -> None:
    """Reset per-job Groq counters and degradation metadata."""
    state = GroqJobState(
        job_id=job_id or str(uuid.uuid4()),
        max_calls=max_calls if max_calls is not None else _env_max_calls(),
        burst_enabled=_env_allow_groq_burst(),
    )
    _groq_job_state.set(state)
    logger.debug(
        "[HybridLLMService] Groq job reset: job_id=%s max_calls=%d burst=%s",
        state.job_id,
        state.max_calls,
        state.burst_enabled,
    )


def get_groq_call_count() -> int:
    """Get current Groq call count for this request/job."""
    return _groq_job_state.get().calls_used


def get_groq_job_metadata() -> Dict[str, Any]:
    """Get per-job Groq limiter/degradation metadata."""
    state = _groq_job_state.get()
    skipped = sorted(state.skipped_calls, key=lambda t: _DEGRADE_ORDER.get(t, 99))
    return {
        "job_id": state.job_id,
        "groq_calls_used": state.calls_used,
        "groq_calls_max": state.max_calls,
        "allow_groq_burst": state.burst_enabled,
        "degraded_mode": state.degraded_mode,
        "skipped_calls": skipped,
        "fallback_to_local": state.fallback_to_local,
    }


class HybridLLMService:
    """
    Groq-only LLM service.
    Call reset_groq_counter() at the start of each new job/request (for metrics).
    """

    groq_service: Optional[GroqService]
    local_service: Optional[Any]
    _groq_semaphore: asyncio.Semaphore = asyncio.Semaphore(max(1, int(os.getenv("GROQ_MAX_IN_FLIGHT", "4"))))

    def __init__(self) -> None:
        # Primary: Groq
        try:
            self.groq_service = GroqService()
            self.groq_available = True
            logger.info("[HybridLLMService] Groq service available")
        except Exception as e:
            logger.warning(f"[HybridLLMService] Groq service unavailable: {e}")
            self.groq_service = None
            self.groq_available = False

        # Optional local fallback service (if implemented/configured in deployment)
        self.local_service = None
        self.local_available = False
        local_enabled = (os.getenv("LOCAL_LLM_ENABLED", "false") or "").strip().lower() in {"1", "true", "yes", "on"}
        if local_enabled:
            try:
                from app.services.llms.local_service import LocalLLMService  # type: ignore

                self.local_service = LocalLLMService()
                self.local_available = True
                logger.info("[HybridLLMService] Local LLM fallback available")
            except Exception as e:
                logger.warning("[HybridLLMService] Local fallback requested but unavailable: %s", e)

        if not self.groq_available and not self.local_available:
            raise RuntimeError("No LLM service available (Groq and local fallback unavailable)")

    def _mark_degraded_skip(self, call_tag: str) -> Dict[str, Any]:
        state = _groq_job_state.get()
        state.degraded_mode = True
        if call_tag and call_tag not in state.skipped_calls:
            state.skipped_calls.append(call_tag)
        _groq_job_state.set(state)
        logger.warning(
            "[HybridLLMService] Degraded mode active: skipped non-critical call '%s' for job=%s",
            call_tag or "unknown",
            state.job_id,
        )
        return {
            "_llm_error": "skipped_due_to_quota",
            "_degraded_skip": True,
            "degraded_mode": True,
            "call_tag": call_tag,
        }

    def _consume_groq_budget(self) -> tuple[bool, int, int]:
        state = _groq_job_state.get()
        if state.burst_enabled:
            state.calls_used += 1
            _groq_job_state.set(state)
            return True, state.calls_used, state.max_calls
        if state.calls_used >= state.max_calls:
            return False, state.calls_used, state.max_calls
        state.calls_used += 1
        _groq_job_state.set(state)
        return True, state.calls_used, state.max_calls

    async def _invoke_local(
        self,
        prompt: str,
        response_format: str,
        temperature: float | None,
    ) -> Dict[str, Any]:
        if not self.local_available or not self.local_service:
            raise RuntimeError("Local LLM fallback unavailable")
        state = _groq_job_state.get()
        state.fallback_to_local = True
        _groq_job_state.set(state)
        return await self.local_service.ainvoke(prompt, response_format=response_format, temperature=temperature)

    async def ainvoke(
        self,
        prompt: str,
        response_format: str = "text",
        priority: LLMPriority = LLMPriority.HIGH,
        temperature: float | None = None,
        call_tag: str = "",
    ) -> Dict[str, Any]:
        """
        Calls LLM with priority-based routing and automatic fallback.

        Args:
            prompt: The prompt to send to the LLM
            response_format: "text" or "json"
            priority: HIGH = use Groq (if quota available), LOW = use Local LLM directly
            temperature: Override default temperature (0.0-1.0, lower = more deterministic)

        Strategy:
        - Groq with per-job hard cap (unless ALLOW_GROQ_BURST=true)
        - Local fallback when available
        - If local unavailable and call is non-critical, return degraded skip marker
        """
        can_use_groq, call_num, max_calls = self._consume_groq_budget()

        if not can_use_groq:
            logger.warning(
                "[HybridLLMService] Groq quota exceeded for job=%s (%d/%d), priority=%s, tag=%s",
                _groq_job_state.get().job_id,
                call_num,
                max_calls,
                priority.value,
                call_tag or "unknown",
            )
            if self.local_available and self.local_service:
                logger.info("[HybridLLMService] Falling back to local LLM (quota exceeded)")
                return await self._invoke_local(prompt, response_format, temperature)
            if priority == LLMPriority.LOW:
                return self._mark_degraded_skip(call_tag or "non_critical")
            raise RuntimeError("Groq quota exceeded and no fallback available for critical call")

        # Try Groq first
        if self.groq_available and self.groq_service:
            try:
                logger.info(
                    "[HybridLLMService] Groq call: job=%s call=%d/%d priority=%s tag=%s",
                    _groq_job_state.get().job_id,
                    call_num,
                    max_calls,
                    priority.value,
                    call_tag or "unspecified",
                )
                async with self._groq_semaphore:
                    result = await self.groq_service.ainvoke(
                        prompt, response_format, max_retries=2, temperature=temperature
                    )
                logger.debug(f"[HybridLLMService] Groq succeeded (call {call_num})")
                return result
            except RateLimitError as e:
                logger.warning(f"[HybridLLMService] Groq rate limited: {e}")
            except Exception as e:
                logger.warning(f"[HybridLLMService] Groq failed: {e}")

        if self.local_available and self.local_service:
            logger.info("[HybridLLMService] Falling back to local LLM after Groq failure")
            return await self._invoke_local(prompt, response_format, temperature)

        if priority == LLMPriority.LOW:
            return self._mark_degraded_skip(call_tag or "non_critical")
        raise RuntimeError("LLM unavailable or request failed")
