from __future__ import annotations

import re
from typing import Any, List

EFFICACY_PATTERNS = [
    r"\bcure(s|d)?\b",
    r"\btreat(s|ed|ment)?\b",
    r"\befficacy\b",
    r"\btherap(y|eutic)\b",
    r"\bclinical trial\b",
    r"\brandomi[sz]ed\b",
    r"\bimprove(s|d)? survival\b",
    r"\bremission\b",
]

SAFETY_ONLY_PATTERNS = [
    r"\bsafe\b",
    r"\btolerat(e|ed|ability)\b",
    r"\bpromotes recovery\b",
    r"\bpostoperative\b",
]


def is_efficacy_relevant(text: str) -> bool:
    t = (text or "").lower()
    has_efficacy = any(re.search(p, t) for p in EFFICACY_PATTERNS)
    has_safety_only = any(re.search(p, t) for p in SAFETY_ONLY_PATTERNS)
    if has_safety_only and not has_efficacy:
        return False
    return has_efficacy


def filter_rationale(evidence: List[Any], *, required_polarity: str, semantic_min: float = 0.40) -> List[Any]:
    required = (required_polarity or "").upper()
    out: List[Any] = []
    for ev in evidence:
        if isinstance(ev, dict):
            sem = float(ev.get("semantic") or ev.get("semantic_score") or ev.get("sem_score") or 0.0)
            stance = str(ev.get("stance") or "").upper()
            text = str(ev.get("text") or ev.get("statement") or "")
        else:
            sem = float(getattr(ev, "semantic", 0.0) or 0.0)
            stance = str(getattr(ev, "stance", "")).upper()
            text = str(getattr(ev, "text", ""))
        if sem < semantic_min:
            continue
        if required == "CONTRADICTS" and stance not in {"CONTRADICTS", "CONTRA", "REFUTES"}:
            continue
        if required == "SUPPORTS" and stance not in {"SUPPORTS", "PRO", "ENTAILS"}:
            continue
        if not is_efficacy_relevant(text):
            continue
        out.append(ev)
    return out
