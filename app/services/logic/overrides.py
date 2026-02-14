from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from app.services.logic.claim_strictness import StrictnessProfile
from app.services.logic.evidence_strength import EvidenceStrength


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass
class LogicOverrideResult:
    override_fired: str
    override_reason: str
    verdict: str
    truthfulness_cap_percent: float | None
    confidence_floor: float | None
    confidence_cap: float | None
    key_numbers: Dict[str, float | int | bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _weighted_segment_coverage(statuses: List[str], valid_set: set[str], partial_set: set[str]) -> float:
    if not statuses:
        return 0.0
    total = len(statuses)
    score = 0.0
    for s in statuses:
        su = str(s or "UNKNOWN").upper()
        if su in valid_set:
            score += 1.0
        elif su in partial_set:
            score += 0.5
    return score / max(1, total)


def _mean(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _confidence_cap(
    base_confidence: float,
    diversity: float,
    evidence_count: int,
    agreement: float,
    strictness: StrictnessProfile,
) -> float:
    cap = 0.98
    low = float(os.getenv("DIVERSITY_CONFIDENCE_CAP_LOW", "0.70"))
    mid = float(os.getenv("DIVERSITY_CONFIDENCE_CAP_MID", "0.80"))
    min_evidence = int(os.getenv("STRICTNESS_MIN_EVIDENCE_COUNT", "3"))
    low_div_thr = float(os.getenv("STRICTNESS_DIVERSITY_LOW_THRESHOLD", "0.10"))
    mid_div_thr = float(os.getenv("STRICTNESS_DIVERSITY_MID_THRESHOLD", "0.20"))
    agreement_min = float(os.getenv("STRICTNESS_AGREEMENT_MIN", "0.70"))

    if diversity < low_div_thr:
        cap = min(cap, low)
    elif diversity < mid_div_thr:
        cap = min(cap, mid)
    if evidence_count < min_evidence:
        cap = min(cap, float(os.getenv("STRICTNESS_LOW_EVIDENCE_CONFIDENCE_CAP", "0.75")))
    if agreement < agreement_min:
        cap = min(cap, float(os.getenv("STRICTNESS_LOW_AGREEMENT_CONFIDENCE_CAP", "0.75")))
    if strictness.required_evidence_level in {"HIGH", "VERY_HIGH"} and diversity < mid_div_thr:
        cap = min(cap, float(os.getenv("STRICTNESS_HIGH_CLAIM_LOW_DIVERSITY_CAP", "0.72")))
    return max(0.05, min(cap, max(0.0, float(base_confidence or 0.0))))


def apply_claim_logic_overrides(
    *,
    claim: str,
    strictness: StrictnessProfile,
    evidence_strengths: List[EvidenceStrength],
    claim_breakdown: List[Dict[str, Any]],
    verdict: str,
    truthfulness_percent: float,
    confidence: float,
    diversity: float,
    agreement: float,
    evidence_count: int,
    kg_hint_ratio: float = 0.0,
) -> LogicOverrideResult:
    statuses = [str((seg or {}).get("status") or "UNKNOWN").upper() for seg in (claim_breakdown or [])]
    support_cov = _weighted_segment_coverage(
        statuses,
        {"VALID", "STRONGLY_VALID"},
        {"PARTIALLY_VALID"},
    )
    refute_cov = _weighted_segment_coverage(statuses, {"INVALID"}, {"PARTIALLY_INVALID"})
    unresolved_ratio = _weighted_segment_coverage(statuses, {"UNKNOWN"}, set())

    supports = [e for e in evidence_strengths if e.stance == "SUPPORTS"]
    refutes = [e for e in evidence_strengths if e.stance == "REFUTES"]

    # Backfill coverage from evidence strengths when breakdown is weak/unknown.
    support_cov = max(
        support_cov, _clamp01(sum(e.support_strength for e in supports) / max(1, len(evidence_strengths)))
    )
    refute_cov = max(refute_cov, _clamp01(sum(e.support_strength for e in refutes) / max(1, len(evidence_strengths))))

    support_hedge = _mean([e.hedge_penalty for e in supports])
    support_rarity = _mean([e.rarity_penalty for e in supports])
    support_strength = _mean([e.support_strength for e in supports])
    refute_strength = _mean([e.support_strength for e in refutes])
    refute_div = 0.0
    if refutes:
        # approximate refute diversity by usable evidence diversity share
        refute_div = _clamp01(min(1.0, (len(refutes) / max(1, evidence_count))) * max(0.0, diversity))

    override = "NONE"
    reason = "none"
    out_verdict = str(verdict or "UNVERIFIABLE").upper()
    truth_cap: float | None = None
    conf_floor: float | None = None

    high_claim = strictness.required_evidence_level in {"HIGH", "VERY_HIGH"}
    refute_cov_force = float(os.getenv("REFUTE_COVERAGE_FORCE_FALSE", "0.50"))
    refute_div_force = float(os.getenv("REFUTE_DIVERSITY_FORCE_FALSE", "0.35"))
    hedge_block = float(os.getenv("HEDGE_PENALTY_BLOCK_TRUE", "0.45"))
    rarity_block = float(os.getenv("RARITY_PENALTY_BLOCK_TRUE", "0.40"))

    if high_claim and refute_cov >= refute_cov_force and refute_div >= refute_div_force:
        override = "CONTRADICTION_DOMINANCE"
        reason = "high-strictness claim with strong diverse refutation coverage"
        out_verdict = "FALSE"
        truth_cap = min(float(truthfulness_percent or 0.0), 20.0)
        conf_floor = 0.55
    elif (
        high_claim
        and out_verdict == "TRUE"
        and (support_hedge >= hedge_block or support_rarity >= rarity_block)
        and (refute_cov >= 0.25 or refute_strength >= 0.45)
    ):
        override = "HEDGE_MISMATCH"
        reason = "broad claim supported mostly by hedged/rare evidence with non-trivial refutation"
        out_verdict = "FALSE" if refute_cov >= 0.45 else "UNVERIFIABLE"
        truth_cap = 20.0 if out_verdict == "FALSE" else 55.0
    elif (
        (strictness.is_conditional or strictness.is_multi_step)
        and out_verdict == "UNVERIFIABLE"
        and support_cov >= 0.45
        and refute_cov < 0.30
    ):
        # Multi-hop relaxation with bounded KG hint influence.
        kg_boost_max = float(os.getenv("MULTIHOP_KG_HINT_MAX_BOOST", "0.15"))
        boosted_support = min(1.0, support_cov + min(max(kg_hint_ratio, 0.0), kg_boost_max))
        if boosted_support >= 0.55:
            override = "MULTIHOP_RELAXATION"
            reason = "conditional/multi-step claim has partial chain support"
            out_verdict = "PARTIALLY_TRUE"
            truth_cap = max(float(truthfulness_percent or 0.0), 45.0)

    capped_conf = _confidence_cap(
        base_confidence=confidence,
        diversity=diversity,
        evidence_count=evidence_count,
        agreement=agreement,
        strictness=strictness,
    )
    cap_fired = capped_conf + 1e-9 < float(confidence or 0.0)
    if override == "NONE" and cap_fired:
        override = "DIVERSITY_CAP"
        reason = "confidence capped due to diversity/evidence/agreement policy"

    return LogicOverrideResult(
        override_fired=override,
        override_reason=reason,
        verdict=out_verdict,
        truthfulness_cap_percent=(round(truth_cap, 3) if truth_cap is not None else None),
        confidence_floor=(round(conf_floor, 3) if conf_floor is not None else None),
        confidence_cap=round(capped_conf, 3),
        key_numbers={
            "support_coverage": round(support_cov, 4),
            "refute_coverage": round(refute_cov, 4),
            "refute_diversity": round(refute_div, 4),
            "unresolved_ratio": round(unresolved_ratio, 4),
            "support_hedge_penalty_mean": round(support_hedge, 4),
            "support_rarity_penalty_mean": round(support_rarity, 4),
            "support_strength_mean": round(support_strength, 4),
            "refute_strength_mean": round(refute_strength, 4),
            "strictness_high": high_claim,
            "kg_hint_ratio": round(float(kg_hint_ratio or 0.0), 4),
            "confidence_cap_applied": bool(cap_fired),
        },
    )
