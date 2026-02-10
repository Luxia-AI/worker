from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

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
    "could",
    "would",
    "should",
    "can",
    "may",
    "might",
    "according",
}

_GENERIC_WEAK_ANCHORS = {
    "do",
    "does",
    "did",
    "not",
    "no",
    "never",
    "cause",
    "causes",
    "caused",
    "cause",
    "link",
    "links",
    "linked",
    "associate",
    "associates",
    "associated",
    "association",
}


def _tokens(text: str) -> List[str]:
    toks = [w for w in re.findall(r"\b[\w\-]+\b", (text or "").lower()) if w and w not in _STOP_WORDS]
    normalized: List[str] = []
    for t in toks:
        n = _lemma_token(t)
        if n:
            normalized.append(n)
    return normalized


def _lemma_token(token: str) -> str:
    t = (token or "").lower().strip()
    if not t:
        return ""
    # Lightweight normalization without external NLP dependencies.
    if len(t) > 5 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 4 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    synonyms = {
        "metabolic": "metabolism",
        "detoxifies": "detox",
        "detoxification": "detox",
        "hydrated": "hydration",
        "hydrate": "hydration",
        "hepatic": "liver",
        "citrus": "lemon",
    }
    return synonyms.get(t, t)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


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


def _infer_polarity(subclaim: str, statement: str) -> str:
    """
    Lightweight polarity alignment:
    - contradiction when negated causal claim is paired with positive/hedged causal statement
    - entails when both are aligned on causal polarity
    """
    sub = (subclaim or "").lower()
    stmt = (statement or "").lower()
    if not sub or not stmt:
        return "neutral"

    neg_causal_re = re.compile(
        (
            r"\b(?:do|does|did|can|could|may|might|must|should|would|will|is|are|was|were)?\s*"
            r"(?:not|never|no)\s+(?:cause|causes|caused|causing|link(?:ed)?|associate(?:d)?|"
            r"result(?:s|ed|ing)?\s+in|lead(?:s|ing)?\s+to)\b"
        ),
        flags=re.IGNORECASE,
    )
    pos_causal_re = re.compile(
        (
            r"\b(?:cause|causes|caused|causing|contribut(?:e|es|ed|ing)\s+to|"
            r"link(?:ed)?\s+to|associate(?:d)?\s+with|lead(?:s|ing)?\s+to|"
            r"result(?:s|ed|ing)?\s+in)\b"
        ),
        flags=re.IGNORECASE,
    )
    hedge_pos_re = re.compile(
        r"\b(?:may|might|can|could|possibly|suggest(?:s|ed)?)\b.{0,25}\b(?:cause|linked?|associate)\b",
        flags=re.IGNORECASE,
    )
    no_link_re = re.compile(r"\bno\s+link\b|\bnot\s+associated\b", flags=re.IGNORECASE)

    sub_neg = bool(neg_causal_re.search(sub) or no_link_re.search(sub))
    stmt_neg = bool(neg_causal_re.search(stmt) or no_link_re.search(stmt))
    sub_pos = bool(pos_causal_re.search(sub)) and not sub_neg
    stmt_pos = bool(pos_causal_re.search(stmt)) and not stmt_neg
    stmt_hedged_pos = bool(hedge_pos_re.search(stmt)) and not stmt_neg

    if sub_neg and (stmt_pos or stmt_hedged_pos) and not stmt_neg:
        return "contradicts"
    if sub_pos and stmt_neg:
        return "contradicts"
    if sub_neg and stmt_neg:
        return "entails"
    if sub_pos and stmt_pos and not stmt_hedged_pos:
        return "entails"
    return "neutral"


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    low = (text or "").lower()
    return any(t in low for t in terms)


def derive_anchor_groups(text: str) -> List[List[str]]:
    """
    Build concept anchor groups.
    A group is satisfied if any term in that group appears in the evidence text.
    """
    t = (text or "").lower()
    groups: List[List[str]] = []

    if _contains_any(t, ["lemon", "citrus"]):
        groups.append(["lemon", "citrus"])
    if _contains_any(t, ["liver", "hepatic"]):
        groups.append(["liver", "hepatic"])
    if _contains_any(t, ["detox", "detoxification", "detoxifies", "cleanse", "cleansing"]):
        groups.append(["detox", "detoxification", "detoxifies", "cleanse", "cleansing"])
    if _contains_any(t, ["metabolism", "metabolic rate", "metabolic", "energy expenditure"]):
        groups.append(["metabolism", "metabolic", "metabolic rate", "energy expenditure"])
    if _contains_any(t, ["hydration", "hydrate", "hydrated"]):
        groups.append(["hydration", "hydrate", "hydrated", "water intake"])

    # Generic anchors: use independent concept tokens (not one "any-of" bucket).
    # This prevents one token (e.g., "vaccines") from satisfying unrelated subclaims
    # like "autism" and "flu" simultaneously.
    if not groups:
        generic = [w for w in _tokens(t) if len(w) > 2 and w not in _GENERIC_WEAK_ANCHORS][:4]
        seen = set()
        for token in generic:
            if token in seen:
                continue
            seen.add(token)
            groups.append([token])
    return groups


def evaluate_anchor_match(text: str, statement: str) -> Dict[str, Any]:
    groups = derive_anchor_groups(text)
    stmt = (statement or "").lower()
    matched = 0
    for g in groups:
        if any(term in stmt for term in g):
            matched += 1
    required = 0
    if groups:
        required = 1 if len(groups) == 1 else min(2, len(groups))
    ok = matched >= required if required > 0 else True
    return {
        "anchor_groups": groups,
        "anchors_per_subclaim": len(groups),
        "required_groups": required,
        "matched_groups": matched,
        "anchor_overlap": (matched / max(1, len(groups))) if groups else 0.0,
        "anchor_ok": ok,
    }


def _relevance(subclaim: str, evidence: Any) -> Dict[str, Any]:
    stmt = _as_statement(evidence)
    sub_toks = set(_tokens(subclaim))
    stmt_toks = set(_tokens(stmt))
    lexical_overlap = len(sub_toks & stmt_toks) / max(1, len(sub_toks))
    base_sem = _as_score(evidence)
    stance = _as_stance(evidence)
    polarity = _infer_polarity(subclaim, stmt)
    anchor_eval = evaluate_anchor_match(subclaim, stmt)
    anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
    overlap = max(lexical_overlap, anchor_overlap)
    relevance = 0.6 * max(0.0, min(1.0, base_sem)) + 0.4 * max(0.0, min(1.0, overlap))

    if not anchor_eval["anchor_ok"]:
        # Penalize weak anchor alignment, but avoid collapsing to zero.
        relevance = min(relevance, 0.45 if anchor_overlap > 0.0 else 0.25)

    # Polarity-aware adjustment so contradictory statements do not inflate coverage.
    if stance == "contradicts" or polarity == "contradicts":
        relevance *= 0.35
    elif stance == "entails" or polarity == "entails":
        relevance = min(1.0, relevance * 1.05)

    return {
        "relevance_score": max(0.0, min(1.0, relevance)),
        "overlap": overlap,
        "lexical_overlap": lexical_overlap,
        "anchor_overlap": anchor_overlap,
        "anchors_matched": int(anchor_eval["matched_groups"]),
        "anchors_required": int(anchor_eval["required_groups"]),
        "anchors_per_subclaim": int(anchor_eval.get("anchors_per_subclaim", 0)),
        "anchor_ok": bool(anchor_eval["anchor_ok"]),
        "stance": stance,
        "polarity": polarity,
    }


def compute_subclaim_coverage(
    subclaims: Sequence[str],
    evidence_list: Iterable[Any],
    strong_threshold: float = 0.55,
    partial_threshold: float = 0.30,
    partial_weight: float = 0.5,
) -> Dict[str, Any]:
    """
    Shared weighted coverage:
    - STRONGLY_VALID => weight 1.0
    - PARTIALLY_VALID => weight `partial_weight`
    - UNKNOWN => weight 0.0
    """
    subclaims = [s for s in (subclaims or []) if (s or "").strip()]
    evidence = list(evidence_list or [])
    if not subclaims:
        return {"coverage": 0.0, "weighted_covered": 0.0, "details": []}

    details: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    for idx, subclaim in enumerate(subclaims):
        best_idx = -1
        best = {
            "relevance_score": 0.0,
            "overlap": 0.0,
            "anchors_matched": 0,
            "anchors_required": 0,
            "anchor_ok": False,
            "stance": "neutral",
            "polarity": "neutral",
        }
        for ev_idx, ev in enumerate(evidence):
            ev_score = _relevance(subclaim, ev)
            if ev_score["relevance_score"] > best["relevance_score"]:
                best = ev_score
                best_idx = ev_idx

        overlap = float(best.get("overlap", 0.0))
        anchor_hits = int(best.get("anchors_matched", 0))
        contradicted = (
            str(best.get("stance", "neutral")).lower() == "contradicts"
            or str(best.get("polarity", "neutral")).lower() == "contradicts"
        )
        if contradicted and overlap >= 0.18 and anchor_hits >= 1:
            status = "UNKNOWN"
            weight = 0.0
        elif (
            best["relevance_score"] >= strong_threshold
            and overlap >= 0.30
            and anchor_hits >= 1
            and bool(best.get("anchor_ok", False))
        ):
            status = "STRONGLY_VALID"
            weight = 1.0
        elif (
            best["relevance_score"] >= partial_threshold
            and overlap >= 0.18
            and anchor_hits >= 1
            and bool(best.get("anchor_ok", False))
        ):
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
                "overlap": round(best["overlap"], 4),
                "lexical_overlap": round(float(best.get("lexical_overlap", 0.0)), 4),
                "anchor_overlap": round(float(best.get("anchor_overlap", 0.0)), 4),
                "anchors_matched": best["anchors_matched"],
                "anchors_required": best["anchors_required"],
                "anchors_per_subclaim": best.get("anchors_per_subclaim", 0),
                "stance": best.get("stance", "neutral"),
                "polarity": best.get("polarity", "neutral"),
                "contradicted": contradicted,
            }
        )

    coverage = weighted_sum / max(1, len(subclaims))
    return {
        "coverage": max(0.0, min(1.0, coverage)),
        "weighted_covered": weighted_sum,
        "details": details,
        "subclaims": len(subclaims),
        "partial_weight": partial_weight,
        "strong_threshold": strong_threshold,
        "partial_threshold": partial_threshold,
    }
