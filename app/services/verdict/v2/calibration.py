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
        if not payload:
            return score

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

        if features:
            coverage = max(0.0, min(1.0, float(features.get("coverage", 0.0) or 0.0)))
            score *= 0.85 + (0.15 * coverage)
        return max(0.0, min(1.0, score))
