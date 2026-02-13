from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from app.shared.anchor_extraction import AnchorExtractor
from app.shared.trust_config import get_trust_config

_STOP_WORDS = {
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
}

_VERB_HINTS = {
    "work",
    "works",
    "working",
    "treat",
    "treats",
    "treating",
    "prevent",
    "prevents",
    "preventing",
    "reduce",
    "reduces",
    "reducing",
    "contain",
    "contains",
    "containing",
    "cause",
    "causes",
    "causing",
    "against",
}

_REFUTATION_CUES = (
    "does not",
    "do not",
    "not effective",
    "ineffective",
    "cannot",
    "can't",
    "no evidence",
    "doesn't",
    "aren't",
    "isn't",
)

_NEGATION_PATTERNS = (
    r"\bdoes\s+not\b",
    r"\bdo\s+not\b",
    r"\bcannot\b",
    r"\bwithout\b",
    r"\bnever\b",
    r"\bnot\b",
    r"\bno\b",
)

_ANCHOR_EXTRACTOR = AnchorExtractor()


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"\b[\w\-]+\b", (text or "").lower()) if w and w not in _STOP_WORDS]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _has_negation(text: str) -> bool:
    low = (text or "").lower()
    if not low:
        return False
    return any(re.search(pattern, low) for pattern in _NEGATION_PATTERNS)


def _strip_negation(text: str) -> str:
    cleaned = (text or "").lower()
    cleaned = re.sub(r"\bdoes\s+not\b", " ", cleaned)
    cleaned = re.sub(r"\bdo\s+not\b", " ", cleaned)
    cleaned = re.sub(r"\bcannot\b", " ", cleaned)
    cleaned = re.sub(r"\bwithout\b", " ", cleaned)
    cleaned = re.sub(r"\bnever\b", " ", cleaned)
    cleaned = re.sub(r"\bnot\b", " ", cleaned)
    cleaned = re.sub(r"\bno\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _aligned_non_negated(subclaim: str, statement: str) -> bool:
    sub_tokens = set(_tokens(_strip_negation(subclaim)))
    stmt_tokens = set(_tokens(_strip_negation(statement)))
    if not sub_tokens or not stmt_tokens:
        return False
    shared = sub_tokens & stmt_tokens
    if len(shared) >= 2:
        return True
    overlap = len(shared) / max(1, len(sub_tokens))
    return overlap >= 0.50


def _as_statement(evidence: Any) -> str:
    if isinstance(evidence, dict):
        return str(evidence.get("statement") or evidence.get("text") or "")
    return str(getattr(evidence, "statement", "") or "")


def _as_score(evidence: Any) -> float:
    if isinstance(evidence, dict):
        return max(
            _safe_float(evidence.get("final_score"), 0.0),
            _safe_float(evidence.get("score"), 0.0),
            _safe_float(evidence.get("sem_score"), 0.0),
            _safe_float(evidence.get("semantic_score"), 0.0),
            _safe_float(evidence.get("trust"), 0.0),
        )
    return max(
        _safe_float(getattr(evidence, "trust", 0.0), 0.0),
        _safe_float(getattr(evidence, "semantic_score", 0.0), 0.0),
    )


def _as_stance(evidence: Any) -> str:
    if isinstance(evidence, dict):
        return str(evidence.get("stance") or "neutral").lower()
    return str(getattr(evidence, "stance", "neutral") or "neutral").lower()


def _is_kg_candidate(evidence: Any) -> bool:
    if isinstance(evidence, dict):
        candidate_type = str(evidence.get("candidate_type") or "").upper()
        source_type = str(evidence.get("source_type") or "").lower()
        return candidate_type == "KG" or source_type == "kg"
    return str(getattr(evidence, "candidate_type", "") or "").upper() == "KG"


def _contains_anchor(statement: str, anchor: str) -> bool:
    low = (statement or "").lower()
    if not anchor:
        return False
    if " " in anchor:
        return anchor in low
    return bool(re.search(rf"\b{re.escape(anchor)}\b", low))


def _object_tokens(text: str) -> set[str]:
    return {tok for tok in _tokens(text) if len(tok) >= 4 and tok not in _VERB_HINTS}


def _has_explicit_refutation_signal(text: str) -> bool:
    low = (text or "").lower()
    if not low:
        return False
    return any(cue in low for cue in _REFUTATION_CUES)


def derive_anchor_groups(text: str, anchors: Sequence[str] | None = None) -> List[List[str]]:
    """
    Build concept anchor groups from dynamic anchors.
    A group is satisfied if any term in that group appears in evidence text.
    """
    cfg = get_trust_config()
    max_groups = max(2, min(5, int(cfg.coverage_anchors_per_subclaim)))
    anchor_terms = [a.strip().lower() for a in (anchors or []) if str(a).strip()]
    if not anchor_terms:
        anchor_terms = [t for t in _tokens(text) if len(t) >= 3][:max_groups]
    groups: List[List[str]] = []
    for term in anchor_terms:
        if term and [term] not in groups:
            groups.append([term])
        if len(groups) >= max_groups:
            break
    return groups


def evaluate_anchor_match(text: str, statement: str, anchors: Sequence[str] | None = None) -> Dict[str, Any]:
    cfg = get_trust_config()
    groups = derive_anchor_groups(text, anchors=anchors)
    stmt = (statement or "").lower()
    matched = 0
    for group in groups:
        if any(_contains_anchor(stmt, term) for term in group):
            matched += 1
    required = min(len(groups), max(1, int(cfg.coverage_min_anchor_hits))) if groups else 0
    return {
        "anchor_groups": groups,
        "anchors_per_subclaim": len(groups),
        "required_groups": required,
        "matched_groups": matched,
        "anchor_overlap": (matched / max(1, len(groups))) if groups else 0.0,
        "anchor_ok": matched >= required if required > 0 else True,
    }


def _relevance(
    subclaim: str,
    evidence: Any,
    anchors: Sequence[str],
    min_anchor_hits: int,
    semantic_weight: float,
    kg_weight: float,
) -> Dict[str, Any]:
    stmt = _as_statement(evidence)
    stmt_low = stmt.lower()
    sub_toks = set(_tokens(subclaim))
    stmt_toks = set(_tokens(stmt))
    lexical_overlap = len(sub_toks & stmt_toks) / max(1, len(sub_toks))
    semantic = max(0.0, min(1.0, _as_score(evidence)))
    stance = _as_stance(evidence)
    anchor_hits = sum(1 for a in anchors if _contains_anchor(stmt_low, a))
    anchor_overlap = anchor_hits / max(1, len(anchors)) if anchors else 0.0
    anchor_component = min(1.0, anchor_hits / max(1, min_anchor_hits))
    overlap = max(lexical_overlap, anchor_overlap)

    # Prefer semantic relevance, then anchor completeness.
    score = (semantic_weight * semantic) + ((1.0 - semantic_weight) * anchor_component)
    score = max(score, 0.60 * semantic + 0.40 * overlap)

    # Down-weight KG evidence when anchors do not align strongly.
    if _is_kg_candidate(evidence):
        if anchor_hits == 0:
            score *= max(0.0, 1.0 - kg_weight)
        elif anchor_hits < min_anchor_hits:
            score *= 0.85

    contradicted = stance == "contradicts"
    if contradicted:
        score *= 0.30

    return {
        "relevance_score": max(0.0, min(1.0, score)),
        "semantic_score": semantic,
        "overlap": overlap,
        "lexical_overlap": lexical_overlap,
        "anchor_overlap": anchor_overlap,
        "anchors_matched": anchor_hits,
        "anchors_required": min_anchor_hits,
        "anchors_per_subclaim": len(anchors),
        "anchor_ok": anchor_hits >= min_anchor_hits,
        "stance": stance,
        "contradicted": contradicted,
        "is_kg_candidate": _is_kg_candidate(evidence),
    }


def compute_subclaim_coverage(
    subclaims: Sequence[str],
    evidence_list: Iterable[Any],
    strong_threshold: float = 0.55,
    partial_threshold: float = 0.30,
    partial_weight: float = 0.5,
    anchors_by_subclaim: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, Any]:
    """
    Shared weighted coverage:
    - STRONGLY_VALID => weight 1.0
    - PARTIALLY_VALID => weight `partial_weight`
    - UNKNOWN => weight 0.0
    """
    cfg = get_trust_config()
    min_anchor_hits = max(1, int(cfg.coverage_min_anchor_hits))
    semantic_weight = max(0.0, min(1.0, float(cfg.rank_semantic_weight)))
    kg_weight = max(0.0, min(1.0, float(cfg.rank_kg_weight)))
    strong_cutoff = max(strong_threshold, float(cfg.coverage_min_relevance_strong))
    partial_cutoff = max(partial_threshold, float(cfg.coverage_min_relevance_partial))

    subclaims = [s for s in (subclaims or []) if (s or "").strip()]
    evidence = list(evidence_list or [])
    if not subclaims:
        return {"coverage": 0.0, "weighted_covered": 0.0, "details": []}

    if anchors_by_subclaim is None:
        anchor_result = _ANCHOR_EXTRACTOR.extract_for_claim(
            claim=" ".join(subclaims),
            subclaims=subclaims,
            entity_hints=[],
        )
        anchors_by_subclaim = anchor_result.anchors_by_subclaim

    details: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    for idx, subclaim in enumerate(subclaims):
        anchors = [a.lower().strip() for a in (anchors_by_subclaim.get(subclaim, []) if anchors_by_subclaim else [])]
        if not anchors:
            anchors = [group[0] for group in derive_anchor_groups(subclaim)]
        if not anchors:
            tokens = [t for t in _tokens(subclaim) if len(t) >= 3]
            anchors = tokens[: max(1, min_anchor_hits)]

        best_idx = -1
        best = {
            "relevance_score": 0.0,
            "semantic_score": 0.0,
            "overlap": 0.0,
            "lexical_overlap": 0.0,
            "anchor_overlap": 0.0,
            "anchors_matched": 0,
            "anchors_required": min_anchor_hits,
            "anchors_per_subclaim": len(anchors),
            "anchor_ok": False,
            "stance": "neutral",
            "contradicted": False,
            "is_kg_candidate": False,
        }
        best_semantic_candidate_idx = -1
        best_semantic_candidate_score = -1.0

        for ev_idx, ev in enumerate(evidence):
            ev_score = _relevance(
                subclaim=subclaim,
                evidence=ev,
                anchors=anchors,
                min_anchor_hits=min_anchor_hits,
                semantic_weight=semantic_weight,
                kg_weight=kg_weight,
            )
            semantic = float(ev_score.get("semantic_score", 0.0))
            if semantic >= partial_cutoff and semantic > best_semantic_candidate_score:
                best_semantic_candidate_score = semantic
                best_semantic_candidate_idx = ev_idx

            if ev_score["relevance_score"] > best["relevance_score"]:
                best = ev_score
                best_idx = ev_idx

        if best_idx < 0 and best_semantic_candidate_idx >= 0:
            best_idx = best_semantic_candidate_idx
            best = _relevance(
                subclaim=subclaim,
                evidence=evidence[best_semantic_candidate_idx],
                anchors=anchors,
                min_anchor_hits=min_anchor_hits,
                semantic_weight=semantic_weight,
                kg_weight=kg_weight,
            )

        contradicted = bool(best.get("contradicted", False))
        semantic = float(best.get("semantic_score", 0.0))
        anchor_hits = int(best.get("anchors_matched", 0))
        best_statement = _as_statement(evidence[best_idx]) if best_idx >= 0 and best_idx < len(evidence) else ""
        sub_obj = _object_tokens(subclaim)
        stmt_obj = _object_tokens(best_statement)
        object_overlap = bool(sub_obj and stmt_obj and (sub_obj & stmt_obj))
        explicit_refutation = _has_explicit_refutation_signal(best_statement)
        claim_has_negation = _has_negation(subclaim)
        evidence_has_negation = _has_negation(best_statement)
        negation_mismatch = claim_has_negation != evidence_has_negation
        high_semantic_mismatch = (
            semantic >= 0.80
            and negation_mismatch
            and (object_overlap or explicit_refutation)
            and (_aligned_non_negated(subclaim, best_statement) or anchor_hits >= 1)
        )
        if contradicted and not (object_overlap or explicit_refutation):
            contradicted = False
        if high_semantic_mismatch:
            contradicted = True
            status = "INVALID"
            weight = 0.0
        elif contradicted:
            status = "UNKNOWN"
            weight = 0.0
        elif semantic >= strong_cutoff and anchor_hits >= min_anchor_hits:
            status = "STRONGLY_VALID"
            weight = 1.0
        elif semantic >= partial_cutoff and anchor_hits >= 1:
            status = "PARTIALLY_VALID"
            weight = partial_weight
        else:
            status = "UNKNOWN"
            weight = 0.0

        weighted_sum += weight
        details.append(
            {
                "subclaim_id": idx + 1,
                "subclaim": subclaim,
                "status": status,
                "weight": weight,
                "best_evidence_id": best_idx,
                "relevance_score": round(best["relevance_score"], 4),
                "semantic_score": round(float(best.get("semantic_score", 0.0)), 4),
                "overlap": round(best["overlap"], 4),
                "lexical_overlap": round(float(best.get("lexical_overlap", 0.0)), 4),
                "anchor_overlap": round(float(best.get("anchor_overlap", 0.0)), 4),
                "anchors": anchors,
                "anchors_matched": anchor_hits,
                "anchors_required": int(best.get("anchors_required", min_anchor_hits)),
                "anchors_per_subclaim": int(best.get("anchors_per_subclaim", len(anchors))),
                "stance": best.get("stance", "neutral"),
                "contradicted": contradicted,
                "is_kg_candidate": bool(best.get("is_kg_candidate", False)),
            }
        )

    coverage = weighted_sum / max(1, len(subclaims))
    return {
        "coverage": max(0.0, min(1.0, coverage)),
        "weighted_covered": weighted_sum,
        "details": details,
        "subclaims": len(subclaims),
        "partial_weight": partial_weight,
        "strong_threshold": strong_cutoff,
        "partial_threshold": partial_cutoff,
    }
