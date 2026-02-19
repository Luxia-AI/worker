from __future__ import annotations

from typing import Iterable

from app.services.verdict.v2.types import EvidenceSignal, SegmentDecision


def aggregate_segment_signals(
    segment: str,
    signals: Iterable[EvidenceSignal],
    valid_threshold: float = 0.22,
    invalid_threshold: float = 0.22,
    min_evidence_mass: float = 0.15,
) -> SegmentDecision:
    support_acc = 1.0
    contra_acc = 1.0
    evidence_ids = []
    mass = 0.0
    for sig in signals:
        if not sig.admissible:
            continue
        trust = max(0.0, min(1.0, float(sig.trust)))
        support_acc *= 1.0 - (max(0.0, min(1.0, float(sig.support_prob))) * trust)
        contra_acc *= 1.0 - (max(0.0, min(1.0, float(sig.contradiction_prob))) * trust)
        mass += trust
        evidence_ids.append(int(sig.evidence_id))

    support_score = 1.0 - support_acc
    contradiction_score = 1.0 - contra_acc
    if mass < min_evidence_mass:
        status = "UNKNOWN"
    else:
        margin = support_score - contradiction_score
        if margin >= valid_threshold:
            status = "VALID"
        elif margin <= -invalid_threshold:
            status = "INVALID"
        elif margin >= 0.06:
            status = "PARTIALLY_VALID"
        elif margin <= -0.06:
            status = "PARTIALLY_INVALID"
        else:
            status = "UNKNOWN"
    return SegmentDecision(
        segment=segment,
        status=status,
        support_score=max(0.0, min(1.0, support_score)),
        contradiction_score=max(0.0, min(1.0, contradiction_score)),
        evidence_ids=sorted(set(evidence_ids)),
    )
