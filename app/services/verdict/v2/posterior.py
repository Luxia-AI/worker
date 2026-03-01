from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable

from app.services.verdict.v2.types import EvidenceScoreV2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x or 0.0)))


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _safe_entropy(weights: Iterable[float]) -> float:
    vals = [max(0.0, float(v or 0.0)) for v in weights]
    s = sum(vals)
    if s <= 0.0:
        return 1.0
    probs = [v / s for v in vals if v > 0.0]
    if not probs:
        return 1.0
    ent = -sum(p * math.log(max(1e-9, p)) for p in probs)
    max_ent = math.log(max(2, len(probs)))
    return _clamp01(ent / max(1e-9, max_ent))


def _agreement_score(scores: Iterable[EvidenceScoreV2]) -> float:
    by_domain = {}
    for s in scores:
        if not s.admissible or s.weight <= 0.0:
            continue
        dom = str(s.source_domain or "unknown")
        if dom not in by_domain:
            by_domain[dom] = {"support": 0.0, "refute": 0.0}
        by_domain[dom]["support"] += float(s.weight) * float(s.support_score)
        by_domain[dom]["refute"] += float(s.weight) * float(s.contradict_score)
    if not by_domain:
        return 0.0
    labels = []
    for mass in by_domain.values():
        if mass["support"] > mass["refute"] + 1e-9:
            labels.append("support")
        elif mass["refute"] > mass["support"] + 1e-9:
            labels.append("refute")
        else:
            labels.append("neutral")
    c = Counter(labels)
    dominant = max(c.values()) if c else 0
    return _clamp01(dominant / max(1, len(labels)))


def compute_posteriors_v2(
    scores: Iterable[EvidenceScoreV2],
    coverage: float,
    diversity: float,
) -> Dict[str, float]:
    items = list(scores or [])
    admissible = [s for s in items if bool(s.admissible) and float(s.weight or 0.0) > 0.0]
    weights = [float(s.weight or 0.0) for s in admissible]
    support_mass = sum(float(s.weight or 0.0) * float(s.support_score or 0.0) for s in admissible)
    refute_mass = sum(float(s.weight or 0.0) * float(s.contradict_score or 0.0) for s in admissible)
    neutral_mass = sum(float(s.weight or 0.0) * float(s.neutral_score or 0.0) for s in admissible)
    mass_total = max(1e-9, support_mass + refute_mass + neutral_mass)
    support_mass_n = _clamp01(support_mass / mass_total)
    refute_mass_n = _clamp01(refute_mass / mass_total)
    neutral_mass_n = _clamp01(neutral_mass / mass_total)

    retrieval_entropy = _safe_entropy(weights)
    agreement = _agreement_score(admissible)
    margin = _clamp01(abs(support_mass_n - refute_mass_n))

    sufficiency = _sigmoid(
        (1.45 * _clamp01(coverage))
        + (1.10 * _clamp01(diversity))
        + (1.30 * _clamp01(support_mass_n + refute_mass_n))
        - (1.20 * retrieval_entropy)
        - 1.25
    )
    p_true_raw = support_mass_n * sufficiency * max(0.20, agreement)
    p_false_raw = refute_mass_n * sufficiency * max(0.20, agreement)
    p_unv_raw = (1.0 - sufficiency) + (neutral_mass_n * (1.0 - margin))
    s = max(1e-9, p_true_raw + p_false_raw + p_unv_raw)
    p_true = _clamp01(p_true_raw / s)
    p_false = _clamp01(p_false_raw / s)
    p_unv = _clamp01(p_unv_raw / s)
    conf_raw = _clamp01(max(p_true, p_false, p_unv) * (0.5 + 0.5 * sufficiency) * (1.0 - 0.30 * retrieval_entropy))
    return {
        "p_true": p_true,
        "p_false": p_false,
        "p_unverifiable": p_unv,
        "support_mass": support_mass_n,
        "refute_mass": refute_mass_n,
        "neutral_mass": neutral_mass_n,
        "margin": margin,
        "sufficiency": sufficiency,
        "agreement": agreement,
        "retrieval_entropy": retrieval_entropy,
        "confidence_raw": conf_raw,
        "admissibility_rate": _clamp01(len(admissible) / max(1, len(items))),
    }
