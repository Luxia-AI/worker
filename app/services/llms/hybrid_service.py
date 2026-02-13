"""Hybrid LLM service with per-job Groq quota hardening and degradation support."""

from __future__ import annotations

import asyncio
import contextvars
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from app.constants.config import (
    LLM_MAX_TOKENS_DEFAULT,
    LLM_MAX_TOKENS_ENTITY_EXTRACTION,
    LLM_MAX_TOKENS_FACT_EXTRACTION,
    LLM_MAX_TOKENS_QUERY_REFORMULATION,
    LLM_MAX_TOKENS_RELATION_EXTRACTION,
    LLM_MAX_TOKENS_VERDICT_GENERATION,
)
from app.core.config import settings
from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService, RateLimitError

logger = get_logger(__name__)


class LLMPriority(Enum):
    """Priority levels for LLM calls."""

    HIGH = "high"  # Use Groq (critical tasks: query reformulation, initial fact extraction)
    LOW = "low"  # Non-critical calls that may be degraded when budget is constrained


@dataclass
class GroqJobState:
    job_id: str
    max_calls: int
    calls_used: int = 0
    burst_enabled: bool = False
    degraded_mode: bool = False
    skipped_calls: list[str] = field(default_factory=list)
    fact_extraction_override_used: bool = False
    warned_events: set[str] = field(default_factory=set)


_groq_job_state: contextvars.ContextVar[GroqJobState] = contextvars.ContextVar(
    "groq_job_state",
    default=GroqJobState(job_id="default", max_calls=5),
)

_DEGRADE_ORDER = {
    "relation_extraction": 1,
    "entity_extraction": 2,
    "fact_extraction": 3,
}

_CRITICAL_CALL_TAGS = {
    "fact_extraction",
    "entity_extraction",
    "relation_extraction",
    "verdict_generation",
}


def _env_confidence_mode() -> bool:
    value = os.getenv("LUXIA_CONFIDENCE_MODE")
    if value is None:
        return bool(getattr(settings, "LUXIA_CONFIDENCE_MODE", False))
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_allow_groq_burst() -> bool:
    return (os.getenv("ALLOW_GROQ_BURST", "false") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_max_unspecified_low_calls() -> int:
    try:
        return max(1, int(os.getenv("GROQ_MAX_UNSPECIFIED_LOW_PER_JOB", "6")))
    except Exception:
        return 6


def _env_max_calls() -> int:
    if _env_confidence_mode():
        try:
            return max(1, int(os.getenv("GROQ_MAX_CALLS_PER_JOB_CONFIDENCE", "30")))
        except Exception:
            return 30
    try:
        return max(1, int(os.getenv("GROQ_MAX_CALLS_PER_JOB", "5")))
    except Exception:
        return 5


def _env_reserved_verdict_calls() -> int:
    try:
        return max(0, int(os.getenv("GROQ_RESERVED_VERDICT_CALLS", "1")))
    except Exception:
        return 1


def _env_reserved_fact_calls() -> int:
    try:
        return max(0, int(os.getenv("GROQ_RESERVED_FACT_EXTRACTION_CALLS", "1")))
    except Exception:
        return 1


def _reserved_call_budget(max_calls: int) -> tuple[int, int]:
    """
    Compute reserved call slots for critical stages.

    Keep at least one non-reserved slot when budgets are very small so
    query planning can still happen before critical stages.
    """
    safe_max = max(1, int(max_calls or 1))
    reserved_verdict = min(_env_reserved_verdict_calls(), safe_max)
    non_verdict_capacity = max(0, safe_max - reserved_verdict)
    fact_cap = max(0, non_verdict_capacity - 1)
    reserved_fact = min(_env_reserved_fact_calls(), fact_cap)
    return reserved_verdict, reserved_fact


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
    reserved_verdict_calls, reserved_fact_calls = _reserved_call_budget(state.max_calls)
    return {
        "job_id": state.job_id,
        "groq_calls_used": state.calls_used,
        "groq_calls_max": state.max_calls,
        "allow_groq_burst": state.burst_enabled,
        "reserved_verdict_calls": reserved_verdict_calls,
        "reserved_fact_extraction_calls": reserved_fact_calls,
        "reserved_critical_calls_total": reserved_verdict_calls + reserved_fact_calls,
        "degraded_mode": state.degraded_mode,
        "skipped_calls": skipped,
        "confidence_mode": _env_confidence_mode(),
        "fact_extraction_override_used": state.fact_extraction_override_used,
    }


class HybridLLMService:
    """
    Groq-only LLM service.
    Call reset_groq_counter() at the start of each new job/request (for metrics).
    """

    groq_service: Optional[GroqService]
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

        if not self.groq_available:
            raise RuntimeError("No LLM service available (Groq unavailable)")

    def _mark_degraded_skip(self, call_tag: str) -> Dict[str, Any]:
        state = _groq_job_state.get()
        state.degraded_mode = True
        should_warn = False
        if call_tag and call_tag not in state.skipped_calls:
            state.skipped_calls.append(call_tag)
            should_warn = True
        _groq_job_state.set(state)
        if should_warn:
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

    @staticmethod
    def _warn_once(state: GroqJobState, key: str, message: str, *args: Any) -> None:
        if key in state.warned_events:
            return
        state.warned_events.add(key)
        _groq_job_state.set(state)
        logger.warning(message, *args)

    @staticmethod
    def _is_critical_call(call_tag: str) -> bool:
        return (call_tag or "").strip().lower() in _CRITICAL_CALL_TAGS

    def _consume_groq_budget(
        self, call_tag: str, priority: LLMPriority, allow_quota_override: bool = False
    ) -> tuple[bool, int, int, str]:
        state = _groq_job_state.get()
        confidence_mode = _env_confidence_mode()
        normalized_tag = (call_tag or "").strip().lower()

        # Guardrail: avoid long tails of low-priority unnamed calls.
        if priority == LLMPriority.LOW and normalized_tag in {"", "unspecified"}:
            max_unspecified = _env_max_unspecified_low_calls()
            if state.calls_used >= max_unspecified:
                self._warn_once(
                    state,
                    f"unspecified_low_cap:{max_unspecified}",
                    "[HybridLLMService] Capping low-priority unspecified Groq calls for job=%s " "(used=%d, cap=%d)",
                    state.job_id,
                    state.calls_used,
                    max_unspecified,
                )
                return False, state.calls_used, state.max_calls, "unspecified_low_cap"

        if state.calls_used >= state.max_calls:
            if confidence_mode and allow_quota_override and self._is_critical_call(call_tag):
                logger.warning(
                    "[HybridLLMService] Confidence override: allowing critical call beyond hard cap "
                    "for job=%s (used=%d/%d, tag=%s)",
                    state.job_id,
                    state.calls_used,
                    state.max_calls,
                    call_tag or "unknown",
                )
                state.calls_used += 1
                _groq_job_state.set(state)
                return (
                    True,
                    state.calls_used,
                    state.max_calls,
                    "confidence_override_hard_cap",
                )
            return False, state.calls_used, state.max_calls, "hard_cap"

        reserved_for_verdict, reserved_for_fact = _reserved_call_budget(state.max_calls)
        reserve_pool = reserved_for_verdict + reserved_for_fact
        tag = normalized_tag
        is_verdict_call = tag == "verdict_generation"
        is_fact_extraction_call = tag == "fact_extraction"
        remaining = state.max_calls - state.calls_used
        allow_high_critical = (
            priority == LLMPriority.HIGH
            and tag in {"entity_extraction", "query_reformulation"}
            and remaining > reserved_for_verdict
        )
        if (
            not is_verdict_call
            and not is_fact_extraction_call
            and remaining <= reserve_pool
            and not allow_high_critical
        ):
            if confidence_mode and is_fact_extraction_call:
                # unreachable due guard, but keep for defensive safety.
                pass
            self._warn_once(
                state,
                (
                    f"preserve_critical:{priority.value}:{call_tag or 'unknown'}:"
                    f"{reserved_for_verdict}:{reserved_for_fact}"
                ),
                "[HybridLLMService] Preserving critical reserved slots for job=%s "
                "(used=%d/%d, remaining=%d, reserve_pool=%d, verdict_reserved=%d, "
                "fact_reserved=%d, priority=%s, tag=%s)",
                state.job_id,
                state.calls_used,
                state.max_calls,
                remaining,
                reserve_pool,
                reserved_for_verdict,
                reserved_for_fact,
                priority.value,
                call_tag or "unknown",
            )
            reason = "reserved_for_verdict" if reserved_for_fact <= 0 else "reserved_for_critical"
            if (
                confidence_mode
                and allow_quota_override
                and is_fact_extraction_call
                and not state.fact_extraction_override_used
            ):
                logger.warning(
                    "[HybridLLMService] Confidence override: bypassing %s for fact extraction " "job=%s (used=%d/%d)",
                    reason,
                    state.job_id,
                    state.calls_used,
                    state.max_calls,
                )
                state.calls_used += 1
                state.fact_extraction_override_used = True
                _groq_job_state.set(state)
                return (
                    True,
                    state.calls_used,
                    state.max_calls,
                    f"confidence_override_{reason}",
                )
            return False, state.calls_used, state.max_calls, reason

        # Fact extraction can use its own reserved pool, but must preserve verdict slot(s).
        if is_fact_extraction_call and remaining <= reserved_for_verdict:
            if confidence_mode and allow_quota_override and not state.fact_extraction_override_used:
                logger.warning(
                    "[HybridLLMService] Confidence override: bypassing reserved_for_verdict "
                    "for fact extraction job=%s (used=%d/%d)",
                    state.job_id,
                    state.calls_used,
                    state.max_calls,
                )
                state.calls_used += 1
                state.fact_extraction_override_used = True
                _groq_job_state.set(state)
                return (
                    True,
                    state.calls_used,
                    state.max_calls,
                    "confidence_override_reserved_for_verdict",
                )
            self._warn_once(
                state,
                f"preserve_verdict:{call_tag or 'unknown'}:{reserved_for_verdict}",
                "[HybridLLMService] Preserving reserved verdict slot(s) for job=%s "
                "(used=%d/%d, remaining=%d, verdict_reserved=%d, tag=%s)",
                state.job_id,
                state.calls_used,
                state.max_calls,
                remaining,
                reserved_for_verdict,
                call_tag or "unknown",
            )
            return False, state.calls_used, state.max_calls, "reserved_for_verdict"

        state.calls_used += 1
        _groq_job_state.set(state)
        return True, state.calls_used, state.max_calls, "ok"

    @staticmethod
    def _max_tokens_for_call(call_tag: str, response_format: str) -> int:
        tag = (call_tag or "").strip().lower()
        mapping = {
            "entity_extraction": LLM_MAX_TOKENS_ENTITY_EXTRACTION,
            "relation_extraction": LLM_MAX_TOKENS_RELATION_EXTRACTION,
            "fact_extraction": LLM_MAX_TOKENS_FACT_EXTRACTION,
            "query_reformulation": LLM_MAX_TOKENS_QUERY_REFORMULATION,
            "verdict_generation": LLM_MAX_TOKENS_VERDICT_GENERATION,
        }
        base = int(mapping.get(tag, LLM_MAX_TOKENS_DEFAULT))
        if _env_confidence_mode() and tag == "fact_extraction":
            base = max(base, int(os.getenv("CONFIDENCE_FACT_EXTRACTION_MAX_TOKENS", "1500")))
        # JSON payloads need a small floor to avoid malformed truncation on short caps.
        if response_format == "json":
            return max(192, base)
        return max(64, base)

    async def ainvoke(
        self,
        prompt: str,
        response_format: str = "text",
        priority: LLMPriority = LLMPriority.HIGH,
        temperature: float | None = None,
        call_tag: str = "",
        allow_quota_override: bool = False,
    ) -> Dict[str, Any]:
        """
        Calls LLM with priority-based routing and automatic fallback.

        Args:
            prompt: The prompt to send to the LLM
            response_format: "text" or "json"
            priority: HIGH = use Groq (if quota available), LOW = use Local LLM directly
            temperature: Override default temperature (0.0-1.0, lower = more deterministic)

        Strategy:
        - Groq with per-job hard cap and reserved verdict-generation slot
        - Per-call output token caps by call type (TPM protection)
        - If quota unavailable and call is non-critical, return degraded skip marker
        """
        confidence_mode = _env_confidence_mode()
        critical_call = self._is_critical_call(call_tag)
        can_use_groq, call_num, max_calls, budget_reason = self._consume_groq_budget(
            call_tag,
            priority,
            allow_quota_override=allow_quota_override,
        )

        if not can_use_groq:
            state = _groq_job_state.get()
            self._warn_once(
                state,
                f"budget_unavailable:{budget_reason}:{priority.value}:{call_tag or 'unknown'}",
                "[HybridLLMService] Groq budget unavailable for job=%s (%d/%d, reason=%s), priority=%s, tag=%s",
                state.job_id,
                call_num,
                max_calls,
                budget_reason,
                priority.value,
                call_tag or "unknown",
            )
            if confidence_mode and critical_call and self.groq_available and self.groq_service:
                if getattr(self.groq_service, "has_fallback_client", False):
                    logger.warning(
                        "[HybridLLMService] Confidence mode: retrying critical call with fallback credentials "
                        "(budget reason=%s, tag=%s)",
                        budget_reason,
                        call_tag or "unknown",
                    )
                    max_tokens = self._max_tokens_for_call(call_tag, response_format)
                    async with self._groq_semaphore:
                        return await self.groq_service.ainvoke(
                            prompt,
                            response_format,
                            max_retries=1,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            force_client="fallback",
                        )
            if priority == LLMPriority.LOW:
                return self._mark_degraded_skip(call_tag or "non_critical")
            if budget_reason == "reserved_for_verdict":
                raise RuntimeError("Groq budget reserved for verdict generation")
            if budget_reason == "reserved_for_critical":
                raise RuntimeError("Groq budget reserved for fact extraction and verdict generation")
            raise RuntimeError("Groq quota exceeded and no fallback available for critical call")

        # Try Groq first
        if self.groq_available and self.groq_service:
            try:
                max_tokens = self._max_tokens_for_call(call_tag, response_format)
                logger.info(
                    "[HybridLLMService] Groq call: job=%s call=%d/%d priority=%s tag=%s max_tokens=%d key=primary",
                    _groq_job_state.get().job_id,
                    call_num,
                    max_calls,
                    priority.value,
                    call_tag or "unspecified",
                    max_tokens,
                )
                async with self._groq_semaphore:
                    result = await self.groq_service.ainvoke(
                        prompt,
                        response_format,
                        max_retries=1 if confidence_mode and critical_call else 2,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                logger.debug(f"[HybridLLMService] Groq succeeded (call {call_num})")
                return result
            except RateLimitError as e:
                logger.warning(f"[HybridLLMService] Groq rate limited: {e}")
                if confidence_mode and critical_call and getattr(self.groq_service, "has_fallback_client", False):
                    try:
                        logger.warning(
                            "[HybridLLMService] Retrying critical call with fallback credentials after rate limit "
                            "(tag=%s)",
                            call_tag or "unknown",
                        )
                        max_tokens = self._max_tokens_for_call(call_tag, response_format)
                        async with self._groq_semaphore:
                            return await self.groq_service.ainvoke(
                                prompt,
                                response_format,
                                max_retries=1,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                force_client="fallback",
                            )
                    except Exception as fallback_error:
                        logger.warning(
                            "[HybridLLMService] Fallback credential retry failed: %s",
                            fallback_error,
                        )
            except Exception as e:
                logger.warning(f"[HybridLLMService] Groq failed: {e}")
                msg = str(e).lower()
                should_retry_fallback = any(
                    token in msg
                    for token in (
                        "quota",
                        "hard_cap",
                        "budget unavailable",
                        "reserved_for",
                        "rate",
                        "timeout",
                        "timed out",
                        "readtimeout",
                    )
                )
                if (
                    confidence_mode
                    and critical_call
                    and should_retry_fallback
                    and getattr(self.groq_service, "has_fallback_client", False)
                ):
                    try:
                        logger.warning(
                            "[HybridLLMService] Retrying critical call with fallback credentials after primary "
                            "failure (tag=%s)",
                            call_tag or "unknown",
                        )
                        max_tokens = self._max_tokens_for_call(call_tag, response_format)
                        async with self._groq_semaphore:
                            return await self.groq_service.ainvoke(
                                prompt,
                                response_format,
                                max_retries=1,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                force_client="fallback",
                            )
                    except Exception as fallback_error:
                        logger.warning(
                            "[HybridLLMService] Fallback credential retry failed: %s",
                            fallback_error,
                        )

        if priority == LLMPriority.LOW:
            return self._mark_degraded_skip(call_tag or "non_critical")
        raise RuntimeError("LLM unavailable or request failed")
