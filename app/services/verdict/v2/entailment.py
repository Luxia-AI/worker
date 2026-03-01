from __future__ import annotations

import math
import os
import re
from typing import Dict, Iterable, List, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _tokenize(text: str) -> List[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "to",
        "for",
        "of",
        "in",
        "on",
        "with",
        "by",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
    }
    return [w for w in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", (text or "").lower()) if w not in stop]


def _negation_flag(text: str) -> bool:
    return bool(re.search(r"\b(no|not|never|without|ineffective|doesn't|dont|do not|does not)\b", text.lower()))


class DeterministicEntailmentVerifier:
    """
    Two-stage contradiction verifier:
    - Optional cross-encoder NLI model (deterministic inference)
    - Conservative heuristic fallback when model is unavailable
    """

    def __init__(self) -> None:
        self._model = None
        self._model_unavailable = False
        self._enabled = os.getenv("REFUTE_NLI_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._model_name = os.getenv("REFUTE_NLI_MODEL", "cross-encoder/nli-deberta-v3-base").strip()

    def _load_model(self) -> bool:
        if not self._enabled or self._model_unavailable:
            return False
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info("[EntailmentV2] Loaded NLI model: %s", self._model_name)
            return True
        except Exception as exc:
            self._model_unavailable = True
            logger.warning("[EntailmentV2] NLI model unavailable, using heuristic fallback: %s", exc)
            return False

    def _heuristic_probs(self, claim: str, evidence: str) -> Tuple[float, float, float]:
        claim_tokens = set(_tokenize(claim))
        evidence_tokens = set(_tokenize(evidence))
        if not claim_tokens or not evidence_tokens:
            return 0.25, 0.25, 0.50

        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        claim_neg = _negation_flag(claim)
        ev_neg = _negation_flag(evidence)
        neg_mismatch = 1.0 if claim_neg != ev_neg else 0.0

        contradict = _sigmoid((1.4 * overlap) + (1.8 * neg_mismatch) - 1.3)
        entail = _sigmoid((1.6 * overlap) + (1.2 * (1.0 - neg_mismatch)) - 1.4)
        neutral = max(0.0, 1.0 - max(entail, contradict))

        s = max(1e-9, entail + contradict + neutral)
        return entail / s, contradict / s, neutral / s

    def score_pair(self, claim: str, evidence: str) -> Dict[str, float]:
        if not claim or not evidence:
            return {"entail": 0.0, "contradict": 0.0, "neutral": 1.0}

        if self._load_model():
            try:
                raw = self._model.predict([(claim, evidence)])
                score = float(raw[0]) if isinstance(raw, (list, tuple)) else float(raw)
                # Map single-score cross-encoder output to a contradiction-sensitive triplet.
                entail = _sigmoid(score)
                h_entail, h_contra, h_neutral = self._heuristic_probs(claim, evidence)
                contradict = max(0.0, min(1.0, (0.65 * h_contra) + (0.35 * (1.0 - entail))))
                neutral = max(0.0, 1.0 - max(entail, contradict))
                s = max(1e-9, entail + contradict + neutral)
                return {"entail": entail / s, "contradict": contradict / s, "neutral": neutral / s}
            except Exception as exc:
                logger.warning("[EntailmentV2] NLI inference failed, fallback heuristic: %s", exc)
        entail, contradict, neutral = self._heuristic_probs(claim, evidence)
        return {"entail": entail, "contradict": contradict, "neutral": neutral}

    def verify_refutes(
        self,
        claim: str,
        candidates: Iterable[Dict[str, object]],
        top_n: int = 5,
    ) -> Dict[int, Dict[str, float]]:
        def _to_id(raw: object) -> int:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return -1

        scored: List[Tuple[float, Dict[str, object]]] = []
        for c in candidates:
            rid = _to_id(c.get("evidence_id", -1))
            if rid < 0:
                continue
            heuristic = float(c.get("stage1_refute_score", 0.0) or 0.0)
            scored.append((heuristic, c))
        scored.sort(key=lambda x: -x[0])
        out: Dict[int, Dict[str, float]] = {}
        for _, c in scored[: max(1, int(top_n))]:
            rid = _to_id(c.get("evidence_id", -1))
            statement = str(c.get("statement") or "")
            probs = self.score_pair(claim, statement)
            out[rid] = probs
        return out
