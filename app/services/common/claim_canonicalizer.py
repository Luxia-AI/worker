from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.logger import get_logger
from app.services.common.claim_segmentation import split_claim_into_segments
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority
from app.services.verdict.v2.entailment import DeterministicEntailmentVerifier

logger = get_logger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).strip(" .,:;")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass(slots=True)
class CanonicalClaimSegment:
    segment_id: str
    original_text: str
    normalized_text: str
    subject: str
    predicate: str
    object: str
    polarity: str
    quantifier: str
    comparator: str
    numeric_value: str
    unit: str
    population: str
    timeframe: str
    modality: str
    parse_confidence: float
    canonical_source: str
    canonical_accepted: bool = True
    canonical_rejected_reason: str = ""
    entail_original_to_canonical: float = 0.0
    entail_canonical_to_original: float = 0.0
    contradiction_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["parse_confidence"] = round(float(self.parse_confidence), 4)
        payload["entail_original_to_canonical"] = round(float(self.entail_original_to_canonical), 4)
        payload["entail_canonical_to_original"] = round(float(self.entail_canonical_to_original), 4)
        payload["contradiction_score"] = round(float(self.contradiction_score), 4)
        return payload


@dataclass(slots=True)
class CanonicalClaim:
    claim_original: str
    segments: List[CanonicalClaimSegment] = field(default_factory=list)

    @property
    def canonical_accept_rate(self) -> float:
        if not self.segments:
            return 0.0
        accepted = sum(1 for s in self.segments if bool(s.canonical_accepted))
        return float(accepted) / float(len(self.segments))

    @property
    def canonical_rejections(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for segment in self.segments:
            if segment.canonical_accepted:
                continue
            out.append(
                {
                    "segment_id": segment.segment_id,
                    "original_text": segment.original_text,
                    "normalized_text": segment.normalized_text,
                    "reason": segment.canonical_rejected_reason,
                    "entail_original_to_canonical": round(float(segment.entail_original_to_canonical), 4),
                    "entail_canonical_to_original": round(float(segment.entail_canonical_to_original), 4),
                    "contradiction_score": round(float(segment.contradiction_score), 4),
                }
            )
        return out

    @property
    def canonical_parse_failures(self) -> List[Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        for segment in self.segments:
            missing_predicate = not bool(str(segment.predicate or "").strip())
            missing_object = not bool(str(segment.object or "").strip())
            low_conf = float(segment.parse_confidence or 0.0) < 0.55
            if not (missing_predicate or missing_object or low_conf):
                continue
            reason_parts: List[str] = []
            if missing_predicate:
                reason_parts.append("missing_predicate")
            if missing_object:
                reason_parts.append("missing_object")
            if low_conf:
                reason_parts.append("low_parse_confidence")
            failures.append(
                {
                    "segment_id": segment.segment_id,
                    "original_text": segment.original_text,
                    "normalized_text": segment.normalized_text,
                    "reason": ",".join(reason_parts),
                    "parse_confidence": round(float(segment.parse_confidence), 4),
                }
            )
        return failures

    def to_dict(self) -> Dict[str, Any]:
        parse_failures = self.canonical_parse_failures
        return {
            "claim_original": self.claim_original,
            "segments": [segment.to_dict() for segment in self.segments],
            "canonical_accept_rate": round(float(self.canonical_accept_rate), 4),
            "canonical_rejections": self.canonical_rejections,
            "canonical_parse_failures": parse_failures,
            "canonical_parse_failed": bool(parse_failures),
            "canonical_failure_reason": "; ".join(str(f.get("reason") or "") for f in parse_failures[:3]),
        }


class ClaimCanonicalizer:
    """
    Dual-track claim canonicalizer.

    - Always keeps original claim/segments.
    - Builds structured canonical segments with deterministic parsing.
    - Optionally uses LLM fallback for low-confidence parses.
    - Guards canonicalized text with bidirectional entailment + contradiction checks.
    """

    _NEGATION_RE = re.compile(
        r"\b(no|not|never|without|ineffective|does not|do not|doesn't|don't|cannot|can't)\b",
        flags=re.IGNORECASE,
    )
    _ABSOLUTE_QUANT_RE = re.compile(r"\b(always|never|all|none|every|only|must)\b", flags=re.IGNORECASE)
    _PROB_QUANT_RE = re.compile(r"\b(may|might|can|could|often|sometimes|typically|generally)\b", flags=re.IGNORECASE)
    _COMPARATOR_RE = re.compile(
        r"\b(more than|less than|fewer than|greater than|higher than|lower than|compared to|versus|vs\.?)\b",
        flags=re.IGNORECASE,
    )
    _PREDICATE_RE = re.compile(
        (
            r"\b("
            r"is|are|was|were|be|being|"
            r"can|cannot|can't|could|should|must|may|might|"
            r"do not|does not|don't|doesn't|"
            r"treat|treats|treated|treating|"
            r"prevent|prevents|prevented|preventing|"
            r"cause|causes|caused|causing|"
            r"reduce|reduces|reduced|reducing|"
            r"increase|increases|increased|increasing|"
            r"improve|improves|improved|improving|"
            r"worsen|worsens|worsened|worsening|"
            r"linked to|associated with|related to|"
            r"required for|necessary for|effective for|effective against"
            r")\b"
        ),
        flags=re.IGNORECASE,
    )
    _NUMERIC_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(%|percent|mg|g|kg|mcg|ml|l|years?|months?|days?|hours?)?\b")
    _TIMEFRAME_RE = re.compile(
        (
            r"\b(per day|daily|weekly|monthly|yearly|"
            r"over \d+ (?:days?|weeks?|months?|years?)|"
            r"in \d+ (?:days?|weeks?|months?|years?))\b"
        ),
        flags=re.IGNORECASE,
    )
    _MODALITY_RE = re.compile(r"\b(can|cannot|can't|may|might|should|must|could|would)\b", flags=re.IGNORECASE)
    _POPULATION_RE = re.compile(
        r"\b(adults?|children|child|infants?|newborns?|pregnant (?:women|people)|women|men|patients?|people|humans?)\b",
        flags=re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.enabled = _env_flag("CLAIM_CANONICALIZATION_ENABLED", True)
        self.llm_fallback_enabled = _env_flag("CLAIM_CANONICAL_LLM_FALLBACK_ENABLED", True)
        self.drift_guard_enabled = _env_flag("CLAIM_CANONICAL_DRIFT_GUARD_ENABLED", True)
        self._entailment = DeterministicEntailmentVerifier()
        self._llm_service: Optional[HybridLLMService] = None

    def _llm(self) -> Optional[HybridLLMService]:
        if not self.llm_fallback_enabled:
            return None
        if self._llm_service is not None:
            return self._llm_service
        try:
            self._llm_service = HybridLLMService()
        except Exception as exc:
            logger.warning("[ClaimCanonicalizer] LLM fallback unavailable: %s", exc)
            self._llm_service = None
        return self._llm_service

    def _split_segments(self, claim: str) -> List[str]:
        segments = split_claim_into_segments(claim)
        cleaned = [_clean_text(seg) for seg in segments if _clean_text(seg)]
        return cleaned if cleaned else [_clean_text(claim)]

    def _rule_parse_segment(self, segment_text: str, segment_id: str) -> CanonicalClaimSegment:
        text = _clean_text(segment_text)
        lower_text = text.lower()

        subject = ""
        predicate = ""
        obj = ""
        predicate_match = None
        for predicate_match in self._PREDICATE_RE.finditer(text):
            break
        if predicate_match:
            predicate = _clean_text(predicate_match.group(0))
            subject = _clean_text(text[: predicate_match.start()])
            obj = _clean_text(text[predicate_match.end() :])
        if subject:
            subject = re.sub(r"\b(?:never|not|no|none|without)\b\s*$", "", subject, flags=re.IGNORECASE).strip()
            subject = re.sub(r"\b(?:has|have|had|is|are|was|were)\b\s*$", "", subject, flags=re.IGNORECASE).strip()
        if not subject:
            tokens = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", text)
            subject = _clean_text(" ".join(tokens[:4])) if tokens else ""

        polarity = "negative" if self._NEGATION_RE.search(lower_text) else "positive"
        quantifier = ""
        if self._ABSOLUTE_QUANT_RE.search(lower_text):
            quantifier = "absolute"
        elif self._PROB_QUANT_RE.search(lower_text):
            quantifier = "probabilistic"

        comparator = ""
        comp_match = self._COMPARATOR_RE.search(text)
        if comp_match:
            comparator = _clean_text(comp_match.group(1))

        numeric_value = ""
        unit = ""
        num_match = self._NUMERIC_RE.search(text)
        if num_match:
            numeric_value = str(num_match.group(1) or "")
            unit = _clean_text(num_match.group(2) or "")

        population = ""
        pop_match = self._POPULATION_RE.search(text)
        if pop_match:
            population = _clean_text(pop_match.group(0))

        timeframe = ""
        tf_match = self._TIMEFRAME_RE.search(text)
        if tf_match:
            timeframe = _clean_text(tf_match.group(0))

        modality = ""
        mod_match = self._MODALITY_RE.search(text)
        if mod_match:
            modality = _clean_text(mod_match.group(1))

        confidence = 0.30
        if subject:
            confidence += 0.20
        if predicate:
            confidence += 0.25
        if obj:
            confidence += 0.20
        if comparator:
            confidence += 0.05
        if numeric_value:
            confidence += 0.05
        if modality:
            confidence += 0.05
        confidence = max(0.05, min(0.99, confidence))

        normalized_bits = [subject, predicate, obj]
        normalized_text = _clean_text(" ".join(part for part in normalized_bits if part))
        if not normalized_text:
            normalized_text = text

        return CanonicalClaimSegment(
            segment_id=segment_id,
            original_text=text,
            normalized_text=normalized_text,
            subject=subject,
            predicate=predicate,
            object=obj,
            polarity=polarity,
            quantifier=quantifier,
            comparator=comparator,
            numeric_value=numeric_value,
            unit=unit,
            population=population,
            timeframe=timeframe,
            modality=modality,
            parse_confidence=confidence,
            canonical_source="rules",
        )

    async def _llm_rewrite_segment(
        self,
        original_text: str,
        current: CanonicalClaimSegment,
    ) -> Optional[CanonicalClaimSegment]:
        llm = self._llm()
        if llm is None:
            return None
        prompt = (
            "Normalize the health claim segment into canonical structured form without changing meaning.\n"
            "Preserve negation, numbers, units, timeframe, and modality exactly.\n"
            "Do not introduce new entities, mechanisms, or causal direction.\n"
            "If uncertain, keep fields close to original text instead of guessing.\n"
            "Return JSON only.\n"
            "{\n"
            '  "normalized_text": "...",\n'
            '  "subject": "...",\n'
            '  "predicate": "...",\n'
            '  "object": "...",\n'
            '  "polarity": "positive|negative",\n'
            '  "quantifier": "",\n'
            '  "comparator": "",\n'
            '  "numeric_value": "",\n'
            '  "unit": "",\n'
            '  "population": "",\n'
            '  "timeframe": "",\n'
            '  "modality": "",\n'
            '  "parse_confidence": 0.0\n'
            "}\n"
            f"Segment: {original_text}\n"
        )
        try:
            raw = await llm.ainvoke(
                prompt,
                response_format="json",
                priority=LLMPriority.LOW,
                temperature=0.0,
                call_tag="claim_canonicalization",
            )
        except Exception as exc:
            logger.warning("[ClaimCanonicalizer] LLM canonicalization failed: %s", exc)
            return None

        if not isinstance(raw, dict):
            return None

        normalized_text = _clean_text(raw.get("normalized_text") or current.normalized_text or original_text)
        subject = _clean_text(raw.get("subject") or current.subject)
        predicate = _clean_text(raw.get("predicate") or current.predicate)
        obj = _clean_text(raw.get("object") or current.object)
        polarity = _clean_text(raw.get("polarity") or current.polarity or "positive").lower()
        quantifier = _clean_text(raw.get("quantifier") or current.quantifier)
        comparator = _clean_text(raw.get("comparator") or current.comparator)
        numeric_value = _clean_text(raw.get("numeric_value") or current.numeric_value)
        unit = _clean_text(raw.get("unit") or current.unit)
        population = _clean_text(raw.get("population") or current.population)
        timeframe = _clean_text(raw.get("timeframe") or current.timeframe)
        modality = _clean_text(raw.get("modality") or current.modality)
        parse_confidence = max(
            0.05, min(0.99, _to_float(raw.get("parse_confidence"), default=current.parse_confidence))
        )

        return CanonicalClaimSegment(
            segment_id=current.segment_id,
            original_text=current.original_text,
            normalized_text=normalized_text,
            subject=subject,
            predicate=predicate,
            object=obj,
            polarity=polarity if polarity in {"positive", "negative"} else current.polarity,
            quantifier=quantifier,
            comparator=comparator,
            numeric_value=numeric_value,
            unit=unit,
            population=population,
            timeframe=timeframe,
            modality=modality,
            parse_confidence=parse_confidence,
            canonical_source="llm",
        )

    def _drift_guard(self, segment: CanonicalClaimSegment) -> CanonicalClaimSegment:
        original = _clean_text(segment.original_text)
        normalized = _clean_text(segment.normalized_text)
        if not original:
            segment.canonical_accepted = False
            segment.canonical_rejected_reason = "empty_original_segment"
            return segment
        if not normalized:
            segment.canonical_accepted = False
            segment.canonical_rejected_reason = "empty_canonical_segment"
            segment.normalized_text = original
            segment.canonical_source = "rules"
            return segment

        e1_probs = self._entailment.score_pair(original, normalized)
        e2_probs = self._entailment.score_pair(normalized, original)
        e1 = _to_float(e1_probs.get("entail"), 0.0)
        e2 = _to_float(e2_probs.get("entail"), 0.0)
        c = max(_to_float(e1_probs.get("contradict"), 0.0), _to_float(e2_probs.get("contradict"), 0.0))

        segment.entail_original_to_canonical = e1
        segment.entail_canonical_to_original = e2
        segment.contradiction_score = c

        original_norm = re.sub(r"[^\w\s]", "", original.lower()).strip()
        canonical_norm = re.sub(r"[^\w\s]", "", normalized.lower()).strip()
        if original_norm and canonical_norm and original_norm == canonical_norm:
            segment.canonical_accepted = True
            segment.canonical_rejected_reason = ""
            return segment

        original_tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", original.lower()))
        canonical_tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", normalized.lower()))
        jaccard = (
            (len(original_tokens & canonical_tokens) / max(1, len(original_tokens | canonical_tokens)))
            if (original_tokens or canonical_tokens)
            else 0.0
        )
        if jaccard >= 0.85 and min(e1, e2) >= 0.50 and c <= 0.35:
            segment.canonical_accepted = True
            segment.canonical_rejected_reason = ""
            return segment

        if min(e1, e2) >= 0.55 and c <= 0.20:
            segment.canonical_accepted = True
            segment.canonical_rejected_reason = ""
            return segment

        segment.canonical_accepted = False
        segment.canonical_rejected_reason = f"drift_guard_failed:min_entail={min(e1, e2):.3f},contradiction={c:.3f}"
        segment.normalized_text = original
        segment.canonical_source = "rules"
        segment.parse_confidence = max(0.05, min(0.99, segment.parse_confidence * 0.8))
        return segment

    async def canonicalize_claim(self, claim: str) -> CanonicalClaim:
        claim_text = _clean_text(claim)
        if not self.enabled:
            segments = self._split_segments(claim_text)
            return CanonicalClaim(
                claim_original=claim_text,
                segments=[
                    CanonicalClaimSegment(
                        segment_id=f"s{i + 1}",
                        original_text=seg,
                        normalized_text=seg,
                        subject="",
                        predicate="",
                        object="",
                        polarity="positive",
                        quantifier="",
                        comparator="",
                        numeric_value="",
                        unit="",
                        population="",
                        timeframe="",
                        modality="",
                        parse_confidence=0.0,
                        canonical_source="rules",
                        canonical_accepted=False,
                        canonical_rejected_reason="canonicalization_disabled",
                    )
                    for i, seg in enumerate(segments)
                ],
            )

        segments_raw = self._split_segments(claim_text)
        parsed_segments: List[CanonicalClaimSegment] = []
        for idx, segment_text in enumerate(segments_raw, start=1):
            segment_id = f"s{idx}"
            parsed = self._rule_parse_segment(segment_text, segment_id=segment_id)
            requires_llm = parsed.parse_confidence < 0.55 or (not parsed.predicate) or (not parsed.object)
            if requires_llm and self.llm_fallback_enabled:
                llm_variant = await self._llm_rewrite_segment(segment_text, parsed)
                if llm_variant is not None:
                    parsed = llm_variant

            if self.drift_guard_enabled:
                parsed = self._drift_guard(parsed)
            if (not parsed.predicate or not parsed.object) and parsed.canonical_accepted:
                parsed.canonical_accepted = False
                parsed.canonical_rejected_reason = parsed.canonical_rejected_reason or "canonical_parse_incomplete"
                parsed.parse_confidence = max(0.05, min(0.99, float(parsed.parse_confidence or 0.0) * 0.80))
            parsed_segments.append(parsed)

        return CanonicalClaim(claim_original=claim_text, segments=parsed_segments)

    @staticmethod
    def split_query_tracks(
        canonical_claim: CanonicalClaim,
    ) -> Tuple[List[str], List[str]]:
        original_track: List[str] = []
        canonical_track: List[str] = []
        for seg in canonical_claim.segments:
            if seg.original_text:
                original_track.append(seg.original_text)
            if seg.canonical_accepted and seg.normalized_text:
                canonical_track.append(seg.normalized_text)
        return original_track, canonical_track

    @staticmethod
    def build_dual_track_queries(
        canonical_claim: CanonicalClaim,
        max_per_segment: int = 8,
    ) -> Dict[str, List[str]]:
        """
        Build bounded dual-track query templates per segment.

        Budget per segment (max 8):
        - 3 original support
        - 2 original refute
        - 2 canonical support
        - 1 canonical refute
        """
        queries_original: List[str] = []
        queries_canonical: List[str] = []
        query_facets: List[Dict[str, str]] = []

        for segment in canonical_claim.segments:
            original = _clean_text(segment.original_text)
            canonical = _clean_text(segment.normalized_text)
            subject = _clean_text(segment.subject)
            predicate = _clean_text(segment.predicate)
            obj = _clean_text(segment.object)
            triple = _clean_text(f"{subject} {predicate} {obj}")

            original_support: List[str] = []
            original_refute: List[str] = []
            canonical_support: List[str] = []
            canonical_refute: List[str] = []

            if original:
                original_support.extend(
                    [
                        original,
                        f"{original} evidence",
                        f"{original} systematic review",
                        f"{original} guideline statement",
                    ]
                )
                if triple:
                    original_refute.extend(
                        [
                            f"no evidence {triple}",
                            f"{subject} does not {predicate} {obj}".strip(),
                        ]
                    )
                else:
                    original_refute.extend(
                        [
                            f"no evidence {original}",
                            f"{original} ineffective",
                        ]
                    )

            if segment.canonical_accepted and canonical:
                canonical_support.extend(
                    [
                        canonical,
                        f"{canonical} clinical evidence",
                        f"{canonical} authoritative guideline",
                    ]
                )
                if triple:
                    canonical_refute.append(f"evidence against {triple}")
                else:
                    canonical_refute.append(f"evidence against {canonical}")
            authority_queries = []
            anchor_query = canonical or original
            if anchor_query:
                authority_queries.extend(
                    [
                        f"{anchor_query} site:who.int OR site:cdc.gov",
                        f"{anchor_query} site:nih.gov OR site:pubmed.ncbi.nlm.nih.gov",
                    ]
                )

            per_segment = (
                original_support[:3]
                + original_refute[:2]
                + canonical_support[:2]
                + canonical_refute[:1]
                + authority_queries[:1]
            )[: max(1, int(max_per_segment))]
            for query in per_segment:
                q = _clean_text(query)
                if not q:
                    continue
                if query in original_support or query in original_refute:
                    if q not in queries_original:
                        queries_original.append(q)
                    facet = "support" if query in original_support else "refute"
                    query_facets.append(
                        {
                            "query": q,
                            "track": "original",
                            "facet": facet,
                            "segment_id": segment.segment_id,
                        }
                    )
                else:
                    if q not in queries_canonical:
                        queries_canonical.append(q)
                    facet = "support"
                    if query in canonical_refute:
                        facet = "refute"
                    elif query in authority_queries:
                        facet = "authoritative"
                    query_facets.append(
                        {
                            "query": q,
                            "track": "canonical",
                            "facet": facet,
                            "segment_id": segment.segment_id,
                        }
                    )

        merged: List[str] = []
        for query in queries_original + queries_canonical:
            if query and query not in merged:
                merged.append(query)
        return {
            "queries_original": queries_original,
            "queries_canonical": queries_canonical,
            "queries_merged": merged,
            "query_facets": query_facets,
        }
