from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

_LEXICON_PATH = Path(__file__).resolve().parent / "lexicons" / "policy_lexicon.json"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _load_lexicon() -> Dict[str, Dict[str, List[str]]]:
    with _LEXICON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


_LEXICON = _load_lexicon()


@dataclass
class StrictnessProfile:
    assertiveness_score: float
    universality_score: float
    modality_score: float
    falsifiability_score: float
    claim_type: str
    required_evidence_level: str
    is_conditional: bool
    is_multi_step: bool
    symptom_inference_score: float

    def to_dict(self) -> Dict[str, float | str | bool]:
        return asdict(self)


def _phrase_hits(text: str, phrases: List[str]) -> int:
    low = f" {text.lower()} "
    return sum(1 for p in phrases if f" {p.lower()} " in low)


def _regex_hits(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _claim_type(claim: str, lx: Dict[str, List[str]]) -> str:
    low = claim.lower()
    if _regex_hits(low, r"\b\d+(\.\d+)?\b") > 0 or "%" in low:
        return "NUMERIC"
    if _phrase_hits(low, lx["comparative_markers"]) > 0:
        return "COMPARATIVE"
    if _phrase_hits(low, lx["causal_connectors"]) > 0:
        if _regex_hits(low, r"\b(risk|odds|likelihood|probability)\b") > 0:
            return "RISK_ASSOCIATION"
        if _regex_hits(low, r"\b(cure|prevent|eliminate|treat|reverse)\b") > 0:
            return "PREVENTION_TREATMENT"
        return "CAUSAL"
    return "FACTUAL"


def compute_claim_strictness(claim_text: str) -> StrictnessProfile:
    claim = str(claim_text or "").strip()
    low = claim.lower()
    lx = _LEXICON.get("claim", {})

    universal = _phrase_hits(low, lx.get("universal_quantifiers", []))
    broad = _phrase_hits(low, lx.get("broad_quantifiers", []))
    tentative = _phrase_hits(low, lx.get("tentative_markers", []))
    definitive = _phrase_hits(low, lx.get("definitive_markers", []))
    causal = _phrase_hits(low, lx.get("causal_connectors", []))
    comparative = _phrase_hits(low, lx.get("comparative_markers", []))
    numeric = _regex_hits(low, r"\b\d+(\.\d+)?\b") + _regex_hits(low, r"\b(mg|g|kg|ml|%|ratio|odds)\b")
    conditional = _phrase_hits(low, lx.get("conditional_markers", []))
    symptom_inf = _phrase_hits(low, lx.get("symptom_inference_markers", []))

    # Assertiveness: strong certainty + causal broad claims.
    assertiveness = _clamp01((1.3 * universal + 0.9 * definitive + 0.7 * causal + 0.4 * broad) / 4.0)
    # Universality: scope of claim, independent from causal language.
    universality = _clamp01((1.2 * universal + 0.6 * broad) / 2.0)
    # Modality: absolute vs probabilistic (higher means absolute).
    modality = _clamp01((1.0 + 0.8 * definitive + 0.7 * universal - 0.8 * tentative - 0.6 * conditional) / 2.0)
    # Falsifiability: claims with numeric/comparative/causal structure are easier to verify/refute.
    falsifiability = _clamp01((0.9 * numeric + 0.8 * comparative + 0.6 * causal + 0.4 * universal) / 3.0)
    if numeric > 0 or comparative > 0:
        falsifiability = _clamp01(falsifiability + 0.08)

    ctype = _claim_type(low, lx)
    is_conditional = conditional > 0 or _regex_hits(low, r"\b(if|when|unless|depending)\b") > 0
    is_multi_step = (
        _regex_hits(low, r"\b(and|then|therefore|through|via)\b") > 0 and len(re.findall(r"[,.;:]", low)) > 0
    )

    strictness_index = _clamp01((0.40 * assertiveness) + (0.30 * universality) + (0.30 * modality))
    if ctype in {"CAUSAL", "PREVENTION_TREATMENT", "NUMERIC", "COMPARATIVE"}:
        strictness_index = _clamp01(strictness_index + 0.08)

    high_thr = float(os.getenv("STRICTNESS_HIGH_THRESHOLD", "0.70"))
    very_high_thr = float(os.getenv("STRICTNESS_VERY_HIGH_THRESHOLD", "0.82"))
    medium_thr = float(os.getenv("STRICTNESS_MEDIUM_THRESHOLD", "0.52"))

    if strictness_index >= very_high_thr:
        level = "VERY_HIGH"
    elif strictness_index >= high_thr:
        level = "HIGH"
    elif strictness_index >= medium_thr:
        level = "MEDIUM"
    else:
        level = "LOW"

    return StrictnessProfile(
        assertiveness_score=round(assertiveness, 4),
        universality_score=round(universality, 4),
        modality_score=round(modality, 4),
        falsifiability_score=round(falsifiability, 4),
        claim_type=ctype,
        required_evidence_level=level,
        is_conditional=is_conditional,
        is_multi_step=is_multi_step,
        symptom_inference_score=round(_clamp01(symptom_inf / 2.0), 4),
    )
