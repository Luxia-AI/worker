from __future__ import annotations

from typing import Any, Dict, Iterable

from app.constants.config import UNVERIFIABLE_CONFIDENCE_CAP
from app.services.verdict.v2.calibration import ConfidenceCalibrator
from app.services.verdict.v2.posterior import compute_posteriors_v2
from app.services.verdict.v2.types import EvidenceScoreV2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x or 0.0)))


def _verdict_from_probs(p_true: float, p_false: float, p_unv: float) -> str:
    scores = {
        "TRUE": float(p_true),
        "FALSE": float(p_false),
        "UNVERIFIABLE": float(p_unv),
    }
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _truthfulness_from_posteriors(verdict: str, p_true: float, p_false: float, p_unv: float) -> float:
    if verdict == "TRUE":
        return min(98.0, max(55.0, 60.0 + (40.0 * _clamp01(p_true))))
    if verdict == "FALSE":
        return min(45.0, max(2.0, 45.0 * (1.0 - _clamp01(p_false))))
    return max(15.0, min(55.0, 35.0 + (15.0 * (_clamp01(p_true) - _clamp01(p_false)))))


def compute_verdict_policy_v2(
    scores: Iterable[EvidenceScoreV2],
    coverage: float,
    diversity: float,
    calibrator: ConfidenceCalibrator,
    calibrator_features: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    post = compute_posteriors_v2(scores=scores, coverage=coverage, diversity=diversity)
    p_true = float(post["p_true"])
    p_false = float(post["p_false"])
    p_unv = float(post["p_unverifiable"])
    verdict = _verdict_from_probs(p_true, p_false, p_unv)

    class_probs_raw = {
        "true": p_true,
        "false": p_false,
        "unverifiable": p_unv,
    }
    features = {
        "coverage": float(coverage or 0.0),
        "admissible_ratio": float(post.get("admissibility_rate", 0.0) or 0.0),
        "contradict_signal": float(post.get("refute_mass", 0.0) or 0.0),
        "support_signal": float(post.get("support_mass", 0.0) or 0.0),
    }
    if calibrator_features:
        features.update({k: float(v) for k, v in calibrator_features.items()})
    class_probs = calibrator.calibrate_distribution(class_probs_raw, features=features)
    class_max = max(float(v or 0.0) for v in class_probs.values())
    confidence_seed = _clamp01((0.65 * float(post.get("confidence_raw", class_max) or class_max)) + (0.35 * class_max))
    calibrated_confidence = calibrator.calibrate(
        confidence_seed,
        features={
            "coverage": features["coverage"],
            "agreement": float(post.get("agreement", 0.0) or 0.0),
            "diversity": float(diversity or 0.0),
            "contradict_signal": features["contradict_signal"],
            "admissible_ratio": features["admissible_ratio"],
            "class_max_prob": class_max,
            "retrieval_depth": 0.0,
        },
    )

    if verdict == "UNVERIFIABLE":
        low_signal = float(post.get("sufficiency", 0.0) or 0.0) < 0.45 or float(post.get("margin", 0.0) or 0.0) < 0.20
        if low_signal:
            calibrated_confidence = min(float(calibrated_confidence), float(UNVERIFIABLE_CONFIDENCE_CAP))
    calibrated_confidence = _clamp01(calibrated_confidence)

    truthfulness = _truthfulness_from_posteriors(verdict, p_true, p_false, p_unv)
    return {
        "verdict": verdict,
        "truthfulness_percent": float(truthfulness),
        "calibrated_confidence": float(calibrated_confidence),
        "class_probs_raw": class_probs_raw,
        "class_probs": class_probs,
        "support_mass": float(post.get("support_mass", 0.0) or 0.0),
        "refute_mass": float(post.get("refute_mass", 0.0) or 0.0),
        "neutral_mass": float(post.get("neutral_mass", 0.0) or 0.0),
        "evidence_sufficiency": float(post.get("sufficiency", 0.0) or 0.0),
        "agreement_score": float(post.get("agreement", 0.0) or 0.0),
        "retrieval_entropy": float(post.get("retrieval_entropy", 0.0) or 0.0),
        "margin": float(post.get("margin", 0.0) or 0.0),
        "admissibility_rate": float(post.get("admissibility_rate", 0.0) or 0.0),
    }
