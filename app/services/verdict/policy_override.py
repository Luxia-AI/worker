from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class OverrideSignals:
    high_grade_support: int
    high_grade_contra: int
    relevant_noncurative: int
    relevant_any: int


def therapeutic_strong_override(sig: OverrideSignals) -> Tuple[str, str]:
    if sig.high_grade_contra >= 1:
        return "FALSE", "high_grade_contradiction"
    if sig.high_grade_support >= 1:
        return "TRUE", "high_grade_support"
    if sig.relevant_noncurative >= 1 or sig.relevant_any >= 1:
        return "MISLEADING", "noncurative_relevant_evidence"
    return "UNVERIFIABLE", "no_relevant_evidence"
