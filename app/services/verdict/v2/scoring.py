from __future__ import annotations

import math
from typing import Dict


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def score_support_contradiction(features: Dict[str, float]) -> tuple[float, float]:
    anchor = float(features.get("anchor_overlap", 0.0) or 0.0)
    predicate = float(features.get("predicate_match", 0.0) or 0.0)
    semantic = float(features.get("semantic_similarity", 0.0) or 0.0)
    trust = float(features.get("trust", 0.0) or 0.0)
    contradiction = float(features.get("contradiction_signal", 0.0) or 0.0)
    intervention = float(features.get("intervention_alignment", 0.0) or 0.0)

    z_support = (
        1.35 * semantic
        + 1.10 * anchor
        + 1.15 * predicate
        + 0.55 * trust
        + 0.40 * intervention
        - 1.85 * contradiction
        - 1.05
    )
    z_contra = (
        1.30 * contradiction
        + 0.70 * predicate
        + 0.35 * trust
        + 0.20 * intervention
        - 0.80 * semantic
        - 0.55 * anchor
        - 0.85
    )
    return _sigmoid(z_support), _sigmoid(z_contra)
