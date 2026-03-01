from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


class ConfidenceCalibrator:
    def __init__(self, calibrator_path: str | None = None) -> None:
        self._path = calibrator_path
        self._payload: Optional[Dict[str, Any]] = None
        self.version: Optional[str] = None
        if calibrator_path:
            self._load()

    def _load(self) -> None:
        try:
            path = Path(self._path or "")
            if not path.exists() or not path.is_file():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._payload = data
                self.version = str(data.get("version") or "v1")
        except Exception:
            self._payload = None
            self.version = None

    def calibrate(self, raw_confidence: float, features: Dict[str, float] | None = None) -> float:
        score = max(0.0, min(1.0, float(raw_confidence or 0.0)))
        payload = self._payload
        if payload:
            mode = str(payload.get("mode") or "linear").lower()
            if mode == "linear":
                slope = float(payload.get("slope", 1.0) or 1.0)
                intercept = float(payload.get("intercept", 0.0) or 0.0)
                score = (slope * score) + intercept
            elif mode == "piecewise":
                knots = payload.get("knots") or []
                if isinstance(knots, list) and knots:
                    chosen_y = None
                    for knot in knots:
                        if not isinstance(knot, dict):
                            continue
                        x = float(knot.get("x", 0.0) or 0.0)
                        y = float(knot.get("y", x) or x)
                        if score <= x:
                            chosen_y = y
                            break
                    if chosen_y is None:
                        chosen_y = float(knots[-1].get("y", score) or score)
                    score = chosen_y
            reliability_bins = payload.get("reliability_bins")
            if isinstance(reliability_bins, list) and reliability_bins:
                mapped = None
                for bucket in reliability_bins:
                    if not isinstance(bucket, dict):
                        continue
                    lower = float(bucket.get("min", 0.0) or 0.0)
                    upper = float(bucket.get("max", 1.0) or 1.0)
                    if lower <= score <= upper:
                        mapped = float(bucket.get("calibrated", score) or score)
                        break
                if mapped is not None:
                    score = mapped

        if features:
            recognized = {
                k
                for k in (
                    "coverage",
                    "agreement",
                    "diversity",
                    "contradict_signal",
                    "admissible_ratio",
                    "evidence_quality",
                    "class_max_prob",
                    "retrieval_depth",
                )
                if k in features
            }
            # Backward-compatible fallback: a single sparse feature should not alter confidence.
            if len(recognized) < 2:
                return max(0.0, min(1.0, score))
            coverage = max(0.0, min(1.0, float(features.get("coverage", 0.0) or 0.0)))
            agreement = max(0.0, min(1.0, float(features.get("agreement", 0.0) or 0.0)))
            diversity = max(0.0, min(1.0, float(features.get("diversity", 0.0) or 0.0)))
            contradict = max(0.0, min(1.0, float(features.get("contradict_signal", 0.0) or 0.0)))
            admissible_ratio = max(0.0, min(1.0, float(features.get("admissible_ratio", 1.0) or 1.0)))
            evidence_quality = max(0.0, min(1.0, float(features.get("evidence_quality", 0.5) or 0.5)))
            class_max_prob = max(0.0, min(1.0, float(features.get("class_max_prob", score) or score)))
            retrieval_depth = max(0.0, min(1.0, float(features.get("retrieval_depth", 0.0) or 0.0)))

            # Reliability-style default calibration even without external payload.
            quality = (
                (0.30 * coverage)
                + (0.25 * agreement)
                + (0.15 * diversity)
                + (0.20 * admissible_ratio)
                + (0.10 * evidence_quality)
            )
            penalty = 0.35 * contradict
            score = (0.50 * score) + (0.25 * quality) + (0.25 * class_max_prob) - penalty
            # Deep retrieval with low evidence quality should not inflate confidence.
            if retrieval_depth >= 0.66 and evidence_quality < 0.45:
                score *= 0.92
            if contradict >= 0.60 and coverage < 0.50:
                score = min(score, 0.62)
            if admissible_ratio < 0.40:
                score *= 0.85
        return max(0.0, min(1.0, score))

    def calibrate_distribution(
        self,
        class_probs: Dict[str, float],
        features: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """
        Class-conditional calibration for three-way verdict probabilities.

        Expected keys: true, false, unverifiable (case-insensitive).
        """
        raw_true = float(class_probs.get("true", class_probs.get("TRUE", 0.0)) or 0.0)
        raw_false = float(class_probs.get("false", class_probs.get("FALSE", 0.0)) or 0.0)
        raw_unv = float(class_probs.get("unverifiable", class_probs.get("UNVERIFIABLE", 0.0)) or 0.0)

        # Normalize safely.
        total = raw_true + raw_false + raw_unv
        if total <= 0.0:
            p_true = p_false = 0.0
            p_unv = 1.0
        else:
            p_true = max(0.0, raw_true / total)
            p_false = max(0.0, raw_false / total)
            p_unv = max(0.0, raw_unv / total)

        payload = self._payload or {}
        # Optional class-conditional temperature scaling from offline artifact.
        if isinstance(payload, dict):
            class_temp = payload.get("class_temperature")
            if isinstance(class_temp, dict):
                t_true = max(1e-3, float(class_temp.get("true", 1.0) or 1.0))
                t_false = max(1e-3, float(class_temp.get("false", 1.0) or 1.0))
                t_unv = max(1e-3, float(class_temp.get("unverifiable", 1.0) or 1.0))

                # Stable logit temperature scaling.
                def _scaled(prob: float, temp: float) -> float:
                    p = max(1e-9, min(1.0 - 1e-9, prob))
                    z = math.log(p / (1.0 - p))
                    s = z / temp
                    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s))))

                p_true = _scaled(p_true, t_true)
                p_false = _scaled(p_false, t_false)
                p_unv = _scaled(p_unv, t_unv)

        if features:
            coverage = max(0.0, min(1.0, float(features.get("coverage", 0.0) or 0.0)))
            admissible_ratio = max(0.0, min(1.0, float(features.get("admissible_ratio", 0.0) or 0.0)))
            contradict = max(0.0, min(1.0, float(features.get("contradict_signal", 0.0) or 0.0)))
            support = max(0.0, min(1.0, float(features.get("support_signal", 0.0) or 0.0)))

            # Push uncertainty up when evidence is sparse or weakly admissible.
            if coverage < 0.45 or admissible_ratio < 0.50:
                p_unv *= 1.18
            # Preserve polarity separation without hard forcing.
            if contradict > support + 0.08:
                p_false *= 1.12
            if support > contradict + 0.08:
                p_true *= 1.10

        # Re-normalize.
        total2 = max(1e-9, p_true + p_false + p_unv)
        p_true /= total2
        p_false /= total2
        p_unv /= total2
        return {
            "true": max(0.0, min(1.0, p_true)),
            "false": max(0.0, min(1.0, p_false)),
            "unverifiable": max(0.0, min(1.0, p_unv)),
        }
