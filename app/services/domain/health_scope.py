from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict


@dataclass
class HealthScopeResult:
    health_in_scope: bool
    biomedical_confidence: float
    scope_reason: str


class HealthScopeGate:
    """
    Lightweight health-domain gate.

    This gate intentionally uses generic biomedical patterns and avoids
    claim- or disease-specific hardcoded rules.
    """

    _PATTERN_WEIGHTS: Dict[str, float] = {
        r"\b(clinical|trial|cohort|case-control|randomized|placebo|double-blind)\b": 0.18,
        r"\b(patient|participants?|subjects?|population|outcome|endpoint)\b": 0.15,
        r"\b(dos(?:e|age)|mg|mcg|g/day|per day|frequency|administration)\b": 0.12,
        r"\b(biomarker|pathway|protein|gene|cellular|receptor|enzyme)\b": 0.12,
        r"\b(symptom|diagnos(?:is|e)|treat(?:ment|ing)?|prevent(?:ion|ive)?|risk)\b": 0.16,
        r"\b(incidence|prevalence|mortality|morbidity|hazard ratio|odds ratio)\b": 0.14,
        r"\b(pubmed|pmc|guideline|systematic review|meta-analysis)\b": 0.13,
    }

    def __init__(self, min_confidence: float = 0.45) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))

    def classify(self, claim_text: str, declared_domain: str | None = None) -> HealthScopeResult:
        text = str(claim_text or "").strip().lower()
        declared = str(declared_domain or "").strip().lower()
        if not text:
            return HealthScopeResult(
                health_in_scope=False,
                biomedical_confidence=0.0,
                scope_reason="empty_claim",
            )

        score = 0.0
        for pattern, weight in self._PATTERN_WEIGHTS.items():
            if re.search(pattern, text):
                score += weight

        # Modest prior boost when caller already routes to health domain.
        if declared in {"health", "biomedical", "medicine"}:
            score += 0.10
        elif declared and declared not in {"general", "unknown"}:
            score -= 0.05

        # Normalize to [0,1].
        confidence = max(0.0, min(1.0, score))
        in_scope = confidence >= self.min_confidence
        reason = "health_scope_detected" if in_scope else "insufficient_health_scope_signal"

        return HealthScopeResult(
            health_in_scope=bool(in_scope),
            biomedical_confidence=round(confidence, 4),
            scope_reason=reason,
        )
