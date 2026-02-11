from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrustConfig:
    coverage_min_anchor_hits: int = 2
    coverage_min_relevance_partial: float = 0.30
    coverage_min_relevance_strong: float = 0.55
    coverage_anchors_per_subclaim: int = 4
    rank_semantic_weight: float = 0.70
    rank_kg_weight: float = 0.30
    search_max_urls_confidence_mode: int = 12
    search_use_llm_query_expansion: bool = True


_CACHE_LOCK = threading.Lock()
_CACHE_VALUE: TrustConfig | None = None
_CACHE_EXPIRES_AT = 0.0
_CACHE_TTL_SECONDS = 60.0


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _defaults() -> Dict[str, Any]:
    return {
        "trust.coverage.min_anchor_hits": _to_int(os.getenv("TRUST_COVERAGE_MIN_ANCHOR_HITS"), 2),
        "trust.coverage.min_relevance_partial": _to_float(os.getenv("TRUST_COVERAGE_MIN_RELEVANCE_PARTIAL"), 0.30),
        "trust.coverage.min_relevance_strong": _to_float(os.getenv("TRUST_COVERAGE_MIN_RELEVANCE_STRONG"), 0.55),
        "trust.coverage.anchors_per_subclaim": _to_int(os.getenv("TRUST_COVERAGE_ANCHORS_PER_SUBCLAIM"), 4),
        "trust.rank.semantic_weight": _to_float(os.getenv("TRUST_RANK_SEMANTIC_WEIGHT"), 0.70),
        "trust.rank.kg_weight": _to_float(os.getenv("TRUST_RANK_KG_WEIGHT"), 0.30),
        "trust.search.max_urls_confidence_mode": _to_int(os.getenv("TRUST_SEARCH_MAX_URLS_CONFIDENCE_MODE"), 12),
        "trust.search.use_llm_query_expansion": _to_bool(
            os.getenv("TRUST_SEARCH_USE_LLM_QUERY_EXPANSION"),
            True,
        ),
    }


def _load_from_azure_app_config() -> Dict[str, Any]:
    endpoint = (
        os.getenv("AZURE_APP_CONFIGURATION_ENDPOINT")
        or os.getenv("APP_CONFIGURATION_ENDPOINT")
        or os.getenv("AZURE_APPCONFIG_ENDPOINT")
    )
    if not endpoint:
        return {}
    try:
        from azure.appconfiguration.provider import SettingSelector, load  # type: ignore
        from azure.identity import DefaultAzureCredential  # type: ignore
    except Exception as exc:
        logger.debug("[trust_config] Azure App Configuration SDK unavailable: %s", exc)
        return {}

    label = os.getenv("AZURE_APP_CONFIGURATION_LABEL")
    selectors = [SettingSelector(key_filter="trust.*", label_filter=label if label else "\0")]
    try:
        config = load(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
            selects=selectors,
        )
    except Exception as exc:
        logger.warning("[trust_config] Failed to read Azure App Configuration: %s", exc)
        return {}

    values: Dict[str, Any] = {}
    for key in (
        "trust.coverage.min_anchor_hits",
        "trust.coverage.min_relevance_partial",
        "trust.coverage.min_relevance_strong",
        "trust.coverage.anchors_per_subclaim",
        "trust.rank.semantic_weight",
        "trust.rank.kg_weight",
        "trust.search.max_urls_confidence_mode",
        "trust.search.use_llm_query_expansion",
    ):
        if key in config:
            values[key] = config.get(key)
    return values


def _build_config(data: Dict[str, Any]) -> TrustConfig:
    semantic_weight = _to_float(data.get("trust.rank.semantic_weight"), 0.70)
    semantic_weight = max(0.0, min(1.0, semantic_weight))
    kg_weight_default = max(0.0, 1.0 - semantic_weight)
    kg_weight = _to_float(data.get("trust.rank.kg_weight"), kg_weight_default)
    kg_weight = max(0.0, min(1.0, kg_weight))
    if semantic_weight + kg_weight == 0.0:
        semantic_weight = 0.70
        kg_weight = 0.30

    return TrustConfig(
        coverage_min_anchor_hits=max(1, _to_int(data.get("trust.coverage.min_anchor_hits"), 2)),
        coverage_min_relevance_partial=max(
            0.0,
            min(1.0, _to_float(data.get("trust.coverage.min_relevance_partial"), 0.30)),
        ),
        coverage_min_relevance_strong=max(
            0.0,
            min(1.0, _to_float(data.get("trust.coverage.min_relevance_strong"), 0.55)),
        ),
        coverage_anchors_per_subclaim=max(1, _to_int(data.get("trust.coverage.anchors_per_subclaim"), 4)),
        rank_semantic_weight=semantic_weight,
        rank_kg_weight=kg_weight,
        search_max_urls_confidence_mode=max(1, _to_int(data.get("trust.search.max_urls_confidence_mode"), 12)),
        search_use_llm_query_expansion=_to_bool(data.get("trust.search.use_llm_query_expansion"), True),
    )


def get_trust_config(force_refresh: bool = False) -> TrustConfig:
    global _CACHE_VALUE, _CACHE_EXPIRES_AT
    now = time.monotonic()
    with _CACHE_LOCK:
        if not force_refresh and _CACHE_VALUE is not None and now < _CACHE_EXPIRES_AT:
            return _CACHE_VALUE

        merged = _defaults()
        merged.update(_load_from_azure_app_config())
        _CACHE_VALUE = _build_config(merged)
        _CACHE_EXPIRES_AT = now + _CACHE_TTL_SECONDS
        return _CACHE_VALUE
