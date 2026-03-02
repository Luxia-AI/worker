from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from app.services.verdict.v2.entailment import DeterministicEntailmentVerifier
from app.services.verdict.v2.normalizer import normalize_relevance_label
from app.services.verdict.v2.types import EvidenceScoreV2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x or 0.0)))


def _domain_from_url(url: str) -> str:
    low = str(url or "").strip().lower()
    if not low:
        return ""
    m = re.match(r"^https?://([^/]+)", low)
    if not m:
        return ""
    dom = m.group(1)
    if dom.startswith("www."):
        dom = dom[4:]
    return dom


def _base_probs_from_relevance(label: str) -> Tuple[float, float, float]:
    if label == "SUPPORTS":
        return 0.72, 0.10, 0.18
    if label == "REFUTES":
        return 0.10, 0.72, 0.18
    return 0.18, 0.18, 0.64


def _stage1_refute_score(claim: str, statement: str, row: Dict[str, Any]) -> float:
    neg_overlap = _clamp01(float(row.get("negation_anchor_overlap", 0.0) or 0.0))
    predicate_match = _clamp01(float(row.get("predicate_match_score", 0.0) or 0.0))
    contradiction_seed = _clamp01(float(row.get("contradiction_score", 0.0) or 0.0))
    has_neg_claim = bool(re.search(r"\b(no|not|never|without|ineffective)\b", (claim or "").lower()))
    has_neg_evidence = bool(re.search(r"\b(no|not|never|without|ineffective)\b", (statement or "").lower()))
    neg_mismatch = 1.0 if has_neg_claim != has_neg_evidence else 0.0
    return _clamp01(
        (0.35 * neg_overlap) + (0.35 * contradiction_seed) + (0.20 * neg_mismatch) + (0.10 * (1.0 - predicate_match))
    )


def _evidence_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def build_evidence_scores_v2(
    claim: str,
    evidence_map: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    nli_top_n: int = 5,
) -> Tuple[List[EvidenceScoreV2], Dict[str, float]]:
    """
    Build deterministic SUPPORT/REFUTE/NEUTRAL evidence scores with two-stage
    contradiction verification.
    """
    verifier = DeterministicEntailmentVerifier()
    stage1_candidates: List[Dict[str, Any]] = []
    indexed_evidence = {int(i): (e or {}) for i, e in enumerate(evidence or [])}

    normalized_rows: List[Dict[str, Any]] = []
    for row in evidence_map or []:
        rid = _evidence_id(row.get("evidence_id", -1))
        if rid < 0:
            continue
        ev = indexed_evidence.get(rid, {})
        statement = str(row.get("statement") or ev.get("statement") or ev.get("text") or "")
        source_url = str(row.get("source_url") or ev.get("source_url") or ev.get("source") or "")
        relevance = normalize_relevance_label(str(row.get("relevance") or "NEUTRAL"))
        stage1 = _stage1_refute_score(claim, statement, row)
        payload = {
            "evidence_id": rid,
            "statement": statement,
            "source_url": source_url,
            "relevance": relevance,
            "relevance_score": _clamp01(float(row.get("relevance_score", 0.0) or 0.0)),
            "credibility": _clamp01(float(row.get("credibility", ev.get("credibility", 0.5)) or 0.5)),
            "scope_alignment": _clamp01(float(row.get("scope_alignment", 1.0) or 1.0)),
            "blocked_content": bool(row.get("blocked_content", False)),
            "nli_entail_prob": _clamp01(float(row.get("nli_entail_prob", 0.0) or 0.0)),
            "nli_contradict_prob": _clamp01(float(row.get("nli_contradict_prob", 0.0) or 0.0)),
            "nli_neutral_prob": _clamp01(float(row.get("nli_neutral_prob", 0.0) or 0.0)),
            "stage1_refute_score": stage1,
        }
        normalized_rows.append(payload)
        if stage1 >= 0.30:
            stage1_candidates.append(payload)

    stage2_verified = verifier.verify_refutes(claim, stage1_candidates, top_n=nli_top_n)

    out: List[EvidenceScoreV2] = []
    refute_verified = 0
    for row in normalized_rows:
        relevance = str(row["relevance"])
        support_b, refute_b, neutral_b = _base_probs_from_relevance(relevance)
        nli_entail = float(row["nli_entail_prob"])
        nli_contra = float(row["nli_contradict_prob"])
        nli_neutral = float(row["nli_neutral_prob"])
        if nli_entail == 0.0 and nli_contra == 0.0 and nli_neutral == 0.0:
            nli_entail, nli_contra, nli_neutral = support_b, refute_b, neutral_b

        verified = stage2_verified.get(int(row["evidence_id"]), {})
        if verified:
            nli_entail = _clamp01(float(verified.get("entail", nli_entail) or nli_entail))
            nli_contra = _clamp01(float(verified.get("contradict", nli_contra) or nli_contra))
            nli_neutral = _clamp01(float(verified.get("neutral", nli_neutral) or nli_neutral))
            if nli_contra >= 0.45:
                refute_verified += 1

        # Directional consistency damping:
        # keep label prior informative when NLI assigns non-trivial opposite mass
        # on clearly directional evidence.
        if relevance == "SUPPORTS" and nli_entail >= nli_contra:
            nli_contra = min(nli_contra, 0.22)
        elif relevance == "REFUTES" and nli_contra >= nli_entail:
            nli_entail = min(nli_entail, 0.22)

        # Blend label prior and NLI posterior.
        support = _clamp01((0.40 * support_b) + (0.60 * nli_entail))
        refute = _clamp01((0.40 * refute_b) + (0.60 * nli_contra))
        neutral = _clamp01((0.40 * neutral_b) + (0.60 * nli_neutral))
        norm = max(1e-9, support + refute + neutral)
        support, refute, neutral = support / norm, refute / norm, neutral / norm

        admissible = bool(not row["blocked_content"] and row["relevance_score"] >= 0.05)
        weight = (
            _clamp01(float(row["relevance_score"]))
            * _clamp01(float(row["credibility"]))
            * _clamp01(float(row["scope_alignment"]))
        )
        if not admissible:
            weight = 0.0

        out.append(
            EvidenceScoreV2(
                evidence_id=int(row["evidence_id"]),
                support_score=float(support),
                contradict_score=float(refute),
                neutral_score=float(neutral),
                nli_entail_prob=float(nli_entail),
                nli_contradict_prob=float(nli_contra),
                nli_neutral_prob=float(nli_neutral),
                admissible=admissible,
                weight=float(weight),
                source_domain=_domain_from_url(str(row["source_url"])),
                metadata={
                    "relevance": relevance,
                    "stage1_refute_score": float(row["stage1_refute_score"]),
                    "source_url": str(row["source_url"]),
                },
            )
        )

    diagnostics = {
        "refute_candidate_count_stage1": float(len(stage1_candidates)),
        "refute_verified_count_stage2": float(refute_verified),
        "refutes_admission_rate": float(refute_verified / max(1, len(out))),
    }
    return out, diagnostics
