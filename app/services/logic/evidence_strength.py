from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.services.ranking.subclaim_coverage import evaluate_anchor_match

_LEXICON_PATH = Path(__file__).resolve().parent / "lexicons" / "policy_lexicon.json"


def _load_lexicon() -> Dict[str, Dict[str, List[str]]]:
    with _LEXICON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


_LEXICON = _load_lexicon()


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _phrase_hits(text: str, phrases: List[str]) -> int:
    low = f" {text.lower()} "
    return sum(1 for p in phrases if f" {p.lower()} " in low)


def _token_overlap(a: str, b: str) -> float:
    stop = {
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
        "that",
        "this",
    }
    sa = {w for w in re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", (a or "").lower()) if w not in stop}
    sb = {w for w in re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", (b or "").lower()) if w not in stop}
    if not sa:
        return 0.0
    return len(sa & sb) / max(1, len(sa))


def _infer_stance(claim: str, text: str, stance_hint: str | None = None) -> str:
    if stance_hint:
        s = str(stance_hint).upper()
        if s in {"CONTRADICTS", "REFUTES", "INVALID", "PARTIALLY_INVALID"}:
            return "REFUTES"
        if s in {"SUPPORTS", "VALID", "PARTIALLY_VALID", "ENTAILS"}:
            return "SUPPORTS"
    c = claim.lower()
    t = text.lower()
    claim_neg = bool(re.search(r"\b(no|not|never|without|none)\b", c))
    text_neg = bool(re.search(r"\b(no|not|never|without|none|does not|cannot|can't)\b", t))
    if claim_neg == text_neg:
        return "SUPPORTS"
    return "REFUTES"


@dataclass
class EvidenceStrength:
    support_strength: float
    hedge_penalty: float
    rarity_penalty: float
    study_quality_hint: float
    stance: str
    negation_anchor_overlap: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


def compute_negation_anchor_overlap(claim_text: str, evidence_text: str) -> float:
    lx = _LEXICON.get("evidence", {})
    claim_l = str(claim_text or "").lower()
    text_l = str(evidence_text or "").lower()
    claim_has_symptom_inference = bool(
        _phrase_hits(claim_l, _LEXICON.get("claim", {}).get("symptom_inference_markers", []))
        or re.search(r"\b(symptom|noticeable|feel|recognize|tell)\b", claim_l)
    )
    if not claim_has_symptom_inference:
        return 0.0
    neg_hits = _phrase_hits(text_l, lx.get("refuting_symptom_negation_markers", []))
    if neg_hits <= 0:
        return 0.0
    anchor = evaluate_anchor_match(claim_text, evidence_text)
    anchor_overlap = float(anchor.get("anchor_overlap", 0.0) or 0.0)
    return _clamp01(0.50 + (0.50 * anchor_overlap))


def compute_evidence_strength(
    claim_text: str,
    text_snippet: str,
    source_meta: Dict[str, Any] | None = None,
    stance_hint: str | None = None,
) -> EvidenceStrength:
    source_meta = source_meta or {}
    text = str(text_snippet or "")
    claim = str(claim_text or "")
    lx = _LEXICON.get("evidence", {})

    hedge_hits = _phrase_hits(text, lx.get("hedge_markers", []))
    uncertainty_hits = _phrase_hits(text, lx.get("uncertainty_markers", []))
    rarity_hits = _phrase_hits(text, lx.get("rarity_markers", []))
    conditional_hits = _phrase_hits(text, lx.get("conditionality_markers", []))
    low_gen_hits = _phrase_hits(text, lx.get("study_context_low_generalizability", []))
    assertive_hits = _phrase_hits(text, lx.get("supporting_assertive_markers", []))

    overlap = _token_overlap(claim, text)
    anchor = evaluate_anchor_match(claim, text)
    anchor_overlap = float(anchor.get("anchor_overlap", 0.0) or 0.0)

    stance = _infer_stance(claim, text, stance_hint=stance_hint)

    hedge_penalty = _clamp01((0.55 * hedge_hits + 0.35 * uncertainty_hits + 0.25 * conditional_hits) / 2.0)
    rarity_penalty = _clamp01((0.65 * rarity_hits + 0.35 * low_gen_hits) / 2.0)

    quality_hint = float(source_meta.get("credibility", 0.5) or 0.5)
    quality_hint = _clamp01(0.70 * quality_hint + 0.30 * _clamp01(assertive_hits / 2.0))

    support_base = (0.45 * overlap) + (0.35 * anchor_overlap) + (0.20 * quality_hint)
    support_strength = _clamp01(support_base - (0.55 * hedge_penalty) - (0.45 * rarity_penalty))

    negation_anchor_overlap = compute_negation_anchor_overlap(claim, text)

    return EvidenceStrength(
        support_strength=round(support_strength, 4),
        hedge_penalty=round(hedge_penalty, 4),
        rarity_penalty=round(rarity_penalty, 4),
        study_quality_hint=round(quality_hint, 4),
        stance=stance,
        negation_anchor_overlap=round(negation_anchor_overlap, 4),
    )
