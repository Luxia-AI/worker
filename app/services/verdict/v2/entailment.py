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


def _softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    clipped = [max(-50.0, min(50.0, float(x))) for x in logits]
    m = max(clipped)
    exps = [math.exp(x - m) for x in clipped]
    total = sum(exps) or 1.0
    return [x / total for x in exps]


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

    def _extract_vector(self, raw: object) -> List[float]:
        # Handle numpy arrays and tensor-like outputs first.
        try:
            import numpy as np

            arr = np.asarray(raw)
            if arr.ndim == 0:
                return [float(arr.item())]
            if arr.ndim == 1:
                return [float(x) for x in arr.tolist()]
            return [float(x) for x in arr[0].tolist()]
        except Exception:
            pass

        # Generic Python fallback.
        if isinstance(raw, (list, tuple)):
            if len(raw) == 1 and isinstance(raw[0], (list, tuple)):
                return [float(x) for x in raw[0]]
            return [float(x) for x in raw]
        return [float(raw)]

    def _label_name_by_index(self) -> Dict[int, str]:
        default = {0: "contradict", 1: "neutral", 2: "entail"}
        try:
            config = getattr(getattr(self._model, "model", None), "config", None)
            id2label = getattr(config, "id2label", None) if config is not None else None
            if not isinstance(id2label, dict) or not id2label:
                return default
            mapped: Dict[int, str] = {}
            for key, value in id2label.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                mapped[idx] = str(value or "").lower()
            return mapped or default
        except Exception:
            return default

    def _probs_from_model_output(self, raw: object) -> Dict[str, float]:
        vec = self._extract_vector(raw)
        if not vec:
            return {}

        # Single score model: interpret as entail logit and blend with heuristic later.
        if len(vec) == 1:
            entail = _sigmoid(vec[0])
            contradict = max(0.0, 1.0 - entail)
            neutral = 0.0
            s = max(1e-9, entail + contradict + neutral)
            return {"entail": entail / s, "contradict": contradict / s, "neutral": neutral / s}

        # Multi-class model (expected: contradiction/neutral/entailment in some order).
        probs = _softmax(vec)
        if len(probs) == 2:
            contradict = probs[0]
            entail = probs[1]
            neutral = max(0.0, 1.0 - max(entail, contradict))
            s = max(1e-9, entail + contradict + neutral)
            return {"entail": entail / s, "contradict": contradict / s, "neutral": neutral / s}

        idx_to_label = self._label_name_by_index()
        entail = 0.0
        contradict = 0.0
        neutral = 0.0
        for idx, prob in enumerate(probs):
            label = idx_to_label.get(idx, "")
            if "entail" in label:
                entail = float(prob)
            elif "contrad" in label or "refut" in label:
                contradict = float(prob)
            elif "neutral" in label:
                neutral = float(prob)

        # Fallback when labels are missing/unexpected: assume standard MNLI order.
        if entail == 0.0 and contradict == 0.0 and neutral == 0.0 and len(probs) >= 3:
            contradict = float(probs[0])
            neutral = float(probs[1])
            entail = float(probs[2])

        s = max(1e-9, entail + contradict + neutral)
        return {"entail": entail / s, "contradict": contradict / s, "neutral": neutral / s}

    def score_pair(self, claim: str, evidence: str) -> Dict[str, float]:
        if not claim or not evidence:
            return {"entail": 0.0, "contradict": 0.0, "neutral": 1.0}

        if self._load_model():
            try:
                raw = self._model.predict([(claim, evidence)])
                h_entail, h_contra, h_neutral = self._heuristic_probs(claim, evidence)
                model_probs = self._probs_from_model_output(raw)
                if model_probs:
                    # Blend model and deterministic heuristic to improve stability.
                    entail = max(
                        0.0,
                        min(1.0, (0.70 * float(model_probs.get("entail", 0.0))) + (0.30 * h_entail)),
                    )
                    contradict = max(
                        0.0,
                        min(1.0, (0.70 * float(model_probs.get("contradict", 0.0))) + (0.30 * h_contra)),
                    )
                    neutral = max(
                        0.0,
                        min(1.0, (0.70 * float(model_probs.get("neutral", 0.0))) + (0.30 * h_neutral)),
                    )
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
