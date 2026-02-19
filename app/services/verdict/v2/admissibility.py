from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class AdmissibilityResult:
    admissible: bool
    reason: str


class EvidenceAdmissibilityGates:
    def __init__(
        self,
        topic_guard: Callable[[str, str], bool],
        claim_admissibility_guard: Callable[[str, str], bool],
        predicate_match: Callable[[str, str], float],
        min_predicate_match: float = 0.15,
    ) -> None:
        self.topic_guard = topic_guard
        self.claim_admissibility_guard = claim_admissibility_guard
        self.predicate_match = predicate_match
        self.min_predicate_match = float(min_predicate_match)

    def check(self, claim_or_segment: str, statement: str, blocked_content: bool = False) -> AdmissibilityResult:
        if blocked_content:
            return AdmissibilityResult(admissible=False, reason="blocked_content")
        if not statement:
            return AdmissibilityResult(admissible=False, reason="empty_statement")
        if not self.topic_guard(claim_or_segment, statement):
            return AdmissibilityResult(admissible=False, reason="topic_guard")
        if not self.claim_admissibility_guard(claim_or_segment, statement):
            return AdmissibilityResult(admissible=False, reason="claim_guard")
        if float(self.predicate_match(claim_or_segment, statement) or 0.0) < self.min_predicate_match:
            return AdmissibilityResult(admissible=False, reason="predicate_guard")
        return AdmissibilityResult(admissible=True, reason="ok")
