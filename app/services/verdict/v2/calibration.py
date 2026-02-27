from __future__ import annotations

import json
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
            coverage = max(0.0, min(1.0, float(features.get("coverage", 0.0) or 0.0)))
            agreement = max(0.0, min(1.0, float(features.get("agreement", 0.0) or 0.0)))
            diversity = max(0.0, min(1.0, float(features.get("diversity", 0.0) or 0.0)))
            contradict = max(0.0, min(1.0, float(features.get("contradict_signal", 0.0) or 0.0)))
            admissible_ratio = max(0.0, min(1.0, float(features.get("admissible_ratio", 1.0) or 1.0)))
            evidence_quality = max(0.0, min(1.0, float(features.get("evidence_quality", 0.5) or 0.5)))

            # Reliability-style default calibration even without external payload.
            quality = (
                (0.30 * coverage)
                + (0.25 * agreement)
                + (0.15 * diversity)
                + (0.20 * admissible_ratio)
                + (0.10 * evidence_quality)
            )
            penalty = 0.35 * contradict
            score = (0.70 * score) + (0.30 * quality) - penalty
            if contradict >= 0.60 and coverage < 0.50:
                score = min(score, 0.62)
            if admissible_ratio < 0.40:
                score *= 0.85
        return max(0.0, min(1.0, score))
