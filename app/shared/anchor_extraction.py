from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from app.core.logger import get_logger
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority
from app.shared.trust_config import get_trust_config

logger = get_logger(__name__)

_ANCHOR_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "to",
    "for",
    "of",
    "in",
    "on",
    "with",
    "by",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "their",
    "there",
    "that",
    "this",
    "these",
    "those",
    "as",
    "about",
    "from",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "must",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "them",
    "my",
    "your",
    "our",
    "his",
    "her",
    "also",
}

_JUNK_ANCHORS = {
    "against",
    "work",
    "works",
    "working",
    "claim",
    "claims",
    "says",
    "said",
    "show",
    "shows",
    "study",
    "studies",
    "medical",
    "health",
    "fact",
    "facts",
}

_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9\-']+\b")
_PUNCT_RE = re.compile(r"[^a-z0-9\-\s]")

_ANCHOR_PROMPT = """Extract 2 to 5 concrete evidence anchors for each subclaim.
Anchors must be biomedical entities or short noun phrases useful for retrieval.
Do not include stopwords or generic verbs.

Return JSON only:
{{"subclaim_anchors":[{{"subclaim":"...","anchors":["...", "..."]}}]}}

Claim:
{claim}

Subclaims:
{subclaims}
"""


@dataclass(slots=True)
class AnchorExtractionResult:
    anchors_by_subclaim: Dict[str, List[str]]
    used_llm: bool


def _normalize_anchor(value: str) -> str:
    text = _PUNCT_RE.sub(" ", (value or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    tokens = [t for t in text.split(" ") if t and t not in _ANCHOR_STOPWORDS and t not in _JUNK_ANCHORS]
    if not tokens:
        return ""
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return ""
    return " ".join(tokens)


def _contains_anchor(text: str, anchor: str) -> bool:
    low = (text or "").lower()
    if not anchor:
        return False
    if " " in anchor:
        return anchor in low
    return bool(re.search(rf"\b{re.escape(anchor)}\b", low))


class AnchorExtractor:
    _shared_llm: HybridLLMService | None = None
    _cache: Dict[str, AnchorExtractionResult] = {}
    _cache_lock = threading.Lock()

    def __init__(self, llm_service: HybridLLMService | None = None) -> None:
        self._llm_service = llm_service

    def _get_llm(self) -> HybridLLMService | None:
        if self._llm_service is not None:
            return self._llm_service
        if os.getenv("ANCHOR_EXTRACTOR_DISABLE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}:
            return None
        if AnchorExtractor._shared_llm is not None:
            self._llm_service = AnchorExtractor._shared_llm
            return self._llm_service
        try:
            AnchorExtractor._shared_llm = HybridLLMService()
            self._llm_service = AnchorExtractor._shared_llm
        except Exception as exc:
            logger.debug("[AnchorExtractor] LLM unavailable, using rule-based anchors: %s", exc)
            self._llm_service = None
        return self._llm_service

    def _sanitize_anchors(self, anchors: Iterable[str], subclaim: str, min_count: int, max_count: int) -> List[str]:
        clean: List[str] = []
        for anchor in anchors:
            normalized = _normalize_anchor(anchor)
            if not normalized:
                continue
            if normalized in clean:
                continue
            if not _contains_anchor(subclaim, normalized):
                # keep high-value biomedical anchors if it is one lexical edit away
                if len(normalized.split()) == 1 and normalized not in (subclaim or "").lower():
                    continue
            clean.append(normalized)
            if len(clean) >= max_count:
                break
        return clean[:max_count] if len(clean) >= min_count else clean

    def _rule_based_anchors(self, subclaim: str, entity_hints: Sequence[str]) -> List[str]:
        cfg = get_trust_config()
        max_count = max(2, min(5, cfg.coverage_anchors_per_subclaim))
        words = [_normalize_anchor(w) for w in _WORD_RE.findall(subclaim or "")]
        words = [w for w in words if w and w not in _JUNK_ANCHORS]
        phrases: List[str] = []
        raw_tokens = [t.lower() for t in _WORD_RE.findall(subclaim or "")]
        for size in (3, 2):
            for idx in range(0, len(raw_tokens) - size + 1):
                span = " ".join(raw_tokens[idx : idx + size])
                norm = _normalize_anchor(span)
                if not norm:
                    continue
                if norm not in phrases:
                    phrases.append(norm)
        hint_candidates: List[str] = []
        for hint in entity_hints or []:
            norm = _normalize_anchor(hint)
            if norm and _contains_anchor(subclaim, norm):
                hint_candidates.append(norm)

        ranked = []
        for candidate in hint_candidates + phrases + words:
            if candidate not in ranked:
                ranked.append(candidate)
        sanitized = self._sanitize_anchors(ranked, subclaim, min_count=2, max_count=max_count)
        if not sanitized:
            fallback_terms = [w for w in words if w and w not in _ANCHOR_STOPWORDS]
            sanitized = fallback_terms[:max_count]
        if len(sanitized) < 2:
            # Never return empty/near-empty anchors.
            tokens = [w for w in words if len(w) >= 3][:2]
            sanitized = (sanitized + tokens)[:max_count]
        return sanitized[:max_count]

    async def _extract_with_llm_async(self, claim: str, subclaims: Sequence[str]) -> Dict[str, List[str]]:
        llm = self._get_llm()
        if llm is None:
            return {}
        prompt = _ANCHOR_PROMPT.format(
            claim=claim,
            subclaims=json.dumps(list(subclaims), ensure_ascii=True),
        )
        result = await llm.ainvoke(
            prompt,
            response_format="json",
            priority=LLMPriority.HIGH,
            call_tag="anchor_extraction",
        )
        payload: Mapping[str, Any]
        if isinstance(result, dict):
            payload = result
        elif isinstance(result, str):
            payload = json.loads(result)
        else:
            return {}
        raw_items: Any = []
        if isinstance(payload, dict):
            for key in ("subclaim_anchors", "anchors", "terms", "keywords"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    raw_items = candidate
                    break
        anchor_map: Dict[str, List[str]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            subclaim = str(item.get("subclaim") or item.get("segment") or item.get("text") or "").strip()
            raw_anchors = (
                item.get("anchors")
                if isinstance(item.get("anchors"), list)
                else (
                    item.get("terms")
                    if isinstance(item.get("terms"), list)
                    else item.get("keywords") if isinstance(item.get("keywords"), list) else []
                )
            )
            if not subclaim or not isinstance(raw_anchors, list):
                continue
            anchor_map[subclaim] = [str(x) for x in raw_anchors if str(x).strip()]
        if not anchor_map and isinstance(payload, dict):
            # Tolerate map-shaped payloads like {"<subclaim>": ["a1", "a2"]}.
            for key, value in payload.items():
                if isinstance(key, str) and isinstance(value, list):
                    cleaned = [str(x) for x in value if str(x).strip()]
                    if cleaned:
                        anchor_map[key.strip()] = cleaned
        return anchor_map

    def _extract_with_llm_sync(self, claim: str, subclaims: Sequence[str]) -> Dict[str, List[str]]:
        result_holder: Dict[str, Any] = {"value": {}}
        err_holder: Dict[str, Exception] = {}

        def _runner() -> None:
            try:
                result_holder["value"] = asyncio.run(self._extract_with_llm_async(claim, subclaims))
            except Exception as exc:
                err_holder["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=8.0)
        if thread.is_alive():
            logger.warning("[AnchorExtractor] LLM anchor extraction timed out; falling back to rule-based anchors")
            return {}
        if "error" in err_holder:
            logger.warning("[AnchorExtractor] LLM anchor extraction failed: %s", err_holder["error"])
            return {}
        value = result_holder.get("value", {})
        return value if isinstance(value, dict) else {}

    def extract_for_claim(
        self,
        claim: str,
        subclaims: Sequence[str],
        entity_hints: Sequence[str] | None = None,
    ) -> AnchorExtractionResult:
        key = "||".join(
            [
                str(claim or "").strip().lower(),
                "|".join(str(s or "").strip().lower() for s in (subclaims or [])),
                "|".join(sorted(str(h or "").strip().lower() for h in (entity_hints or []))),
            ]
        )
        with AnchorExtractor._cache_lock:
            cached = AnchorExtractor._cache.get(key)
        if cached is not None:
            return AnchorExtractionResult(
                anchors_by_subclaim={k: list(v) for k, v in cached.anchors_by_subclaim.items()},
                used_llm=cached.used_llm,
            )

        cfg = get_trust_config()
        max_count = max(2, min(5, cfg.coverage_anchors_per_subclaim))
        hints = [str(h).strip() for h in (entity_hints or []) if str(h).strip()]
        llm_map = self._extract_with_llm_sync(claim, subclaims)
        used_llm = bool(llm_map)

        anchors_by_subclaim: Dict[str, List[str]] = {}
        for subclaim in subclaims:
            normalized_subclaim = str(subclaim or "").strip()
            if not normalized_subclaim:
                continue
            llm_anchors = llm_map.get(normalized_subclaim, [])
            sanitized = self._sanitize_anchors(llm_anchors, normalized_subclaim, min_count=2, max_count=max_count)
            if len(sanitized) < 2:
                sanitized = self._rule_based_anchors(normalized_subclaim, hints)
            if len(sanitized) < 2:
                sanitized = (sanitized + self._rule_based_anchors(normalized_subclaim, []))[:max_count]
            anchors_by_subclaim[normalized_subclaim] = sanitized[:max_count]

        result = AnchorExtractionResult(anchors_by_subclaim=anchors_by_subclaim, used_llm=used_llm)
        with AnchorExtractor._cache_lock:
            AnchorExtractor._cache[key] = result
        return AnchorExtractionResult(
            anchors_by_subclaim={k: list(v) for k, v in result.anchors_by_subclaim.items()},
            used_llm=result.used_llm,
        )
