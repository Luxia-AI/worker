from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.constants.config import (
    CREDIBILITY_AUTHORITY,
    CREDIBILITY_DEFAULT,
    CREDIBILITY_EDU_GOV,
    CREDIBILITY_NEWS,
    RANKING_MIN_CLAIM_OVERLAP,
    RANKING_MIN_CREDIBILITY_THRESHOLD,
    RANKING_MIN_SCORE_FLOOR,
    RANKING_WEIGHTS,
    RECENCY_HALF_LIFE_DAYS,
    TRUSTED_DOMAINS_AUTHORITY,
    TRUSTED_DOMAINS_EDU_GOV,
    TRUSTED_DOMAINS_NEWS,
)
from app.core.logger import get_logger
from app.services.common.list_ops import dedupe_list
from app.services.ranking.subclaim_coverage import evaluate_anchor_match

logger = get_logger(__name__)

LOW_SIGNAL_PHRASES = (
    "data element definitions",
    "registration or results information",
    "javascript and cookies",
    "requires human verification",
    "enable javascript",
)
UNCERTAINTY_PHRASES = (
    "less certain",
    "uncertain",
    "unclear",
    "inconclusive",
    "insufficient evidence",
    "mixed evidence",
)


def _safe_float(v: Optional[float], default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _normalize_scores(values: List[float]) -> List[float]:
    """
    Min-max normalize a list of floats to [0,1]. If all values equal,
    return the values as-is (don't boost single items to 1.0).
    """
    if not values:
        return []

    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        # all values equal: return as-is (preserve original scores)
        return list(values)

    out = [(v - vmin) / (vmax - vmin) for v in values]
    return out


def _recency_boost(published_at: Optional[str], now: Optional[datetime] = None) -> float:
    """
    Convert ISO publish date to a recency boost in [0,1].
    Very recent -> 1.0, old -> approaches 0.
    Uses exponential decay with half-life of 365 days by default.
    """
    if not published_at:
        return 0.0

    try:
        now = now or datetime.now(timezone.utc)
        # accept date-only or full ISO string
        dt = datetime.fromisoformat(published_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = (now - dt).days
        # half-life: 365 days -> weight = 0.5 at 1 year
        decay = 0.5 ** (days / RECENCY_HALF_LIFE_DAYS)
        return float(max(0.0, min(decay, 1.0)))
    except Exception:
        return 0.0


def _entity_overlap_score(query_entities: List[str], item_entities: List[str]) -> float:
    """
    Recall-based overlap: what proportion of query entities appear in item entities.
    Returns 0..1. Perfect match when all query entities found.
    """
    if not query_entities or not item_entities:
        return 0.0

    set_q = {e.strip().lower() for e in query_entities if e}
    set_i = {e.strip().lower() for e in item_entities if e}
    if not set_q or not set_i:
        return 0.0

    inter = set_q.intersection(set_i)
    exact_recall = len(inter) / len(set_q)

    # Token-aware overlap handles phrase/entity granularity mismatch
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
    }
    q_tokens = {w for e in set_q for w in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", e) if w not in stop}
    i_tokens = {w for e in set_i for w in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", e) if w not in stop}
    if not q_tokens or not i_tokens:
        return exact_recall

    token_recall = len(q_tokens & i_tokens) / max(1, len(q_tokens))
    return max(exact_recall, token_recall)


def _claim_overlap_score(claim_text: str, statement: str) -> float:
    """
    Lexical overlap between claim text and evidence statement.
    Returns 0..1 based on proportion of claim content words present in statement.
    """
    if not claim_text or not statement:
        return 0.0

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
        "could",
        "would",
        "should",
        "can",
        "may",
        "might",
        "about",
        "over",
        "under",
        "average",
    }
    claim_words = [w for w in re.findall(r"\b\w+\b", claim_text.lower()) if w not in stop]
    stmt_words = [w for w in re.findall(r"\b\w+\b", statement.lower()) if w not in stop]
    if not claim_words or not stmt_words:
        return 0.0

    claim_set = set(claim_words)
    stmt_set = set(stmt_words)
    overlap = len(claim_set & stmt_set)
    return overlap / max(1, len(claim_set))


def _credibility_score_from_meta(meta: Dict[str, Any]) -> float:
    """
    Return credibility boost only for known authoritative sources.
    Returns a value in [0, 1]: high for trusted domains, 0.5 for known-but-lower-tier
    domains (news/blogs), and 0.0 for unknown sources.
    """

    # explicit field takes precedence
    c = meta.get("credibility")
    if c is not None:
        try:
            return float(c)
        except (ValueError, TypeError):
            # If credibility is not convertible to float, fall through to domain heuristics
            logger.debug(f"[hybrid_rank] Invalid credibility value: {c}, using domain heuristics")

    # domain heuristics
    domain = (meta.get("source_url") or meta.get("source") or "").lower()
    if domain:
        # very small whitelist (customize per deployment)
        if any(d in domain for d in TRUSTED_DOMAINS_AUTHORITY):
            return CREDIBILITY_AUTHORITY
        if any(d in domain for d in TRUSTED_DOMAINS_EDU_GOV):
            return CREDIBILITY_EDU_GOV
        if any(d in domain for d in TRUSTED_DOMAINS_NEWS):
            return CREDIBILITY_NEWS

    # no source or unknown source
    return CREDIBILITY_DEFAULT


def hybrid_rank(
    semantic_results: List[Dict[str, Any]],
    kg_results: List[Dict[str, Any]],
    query_entities: Optional[List[str]] = None,
    query_text: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Combine semantic (vector) and KG results into a single ranked list.

    Inputs:
      - semantic_results: list of dicts with keys:
           - "statement": str
           - "score": float (semantic similarity, higher is better)
           - "entities": List[str] (optional)
           - "source_url" or "source" (optional)
           - "published_at" (optional ISO str)
           - "credibility" (optional)
      - kg_results: list of dicts with keys:
           - "statement": str
           - "score": float (kg score / path quality)
           - "entities": List[str]
           - "source_url" etc.
      - query_entities: entities extracted from the original post (used for overlap)
      - weights: dict of weights (defaults below)

    Returns:
      - list of evidence dicts sorted by final_score desc. Each dict includes:
          - "statement","entities","source_url","published_at","final_score", plus original fields
    """

    query_entities = query_entities or []
    query_text = query_text or ""
    weights = weights or {}
    # weights defaults
    w_sem = float(weights.get("w_semantic", RANKING_WEIGHTS["w_semantic"]))
    w_kg = float(weights.get("w_kg", RANKING_WEIGHTS["w_kg"]))
    w_entity = float(weights.get("w_entity", RANKING_WEIGHTS["w_entity"]))
    w_claim_overlap = float(weights.get("w_claim_overlap", RANKING_WEIGHTS["w_claim_overlap"]))
    w_recency = float(weights.get("w_recency", RANKING_WEIGHTS["w_recency"]))
    w_cred = float(weights.get("w_credibility", RANKING_WEIGHTS["w_credibility"]))

    # Build unified candidate list keyed by (statement, source_url) to merge duplicates
    candidates_map: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    # Collect raw scores for normalization
    sem_scores = []
    kg_scores = []

    # Normalize input fields
    def _add_candidate(src_type: str, item: Dict[str, Any]) -> None:
        stmt = (item.get("statement") or "").strip()
        source_url = item.get("source_url") or item.get("source")
        # canonical key
        key = (stmt, source_url)
        if key not in candidates_map:
            candidates_map[key] = {
                "statement": stmt,
                "source_url": source_url,
                "published_at": item.get("published_at"),
                "entities": item.get("entities", []) or [],
                "sem_score": 0.0,
                "sem_score_raw": 0.0,
                "kg_score": 0.0,
                "kg_score_raw": 0.0,
                "candidate_type": item.get("candidate_type") or ("KG" if src_type == "kg" else "VDB"),
                "is_backfill": bool(item.get("is_backfill", False)),
                "credibility": _credibility_score_from_meta(item),
                "orig": {"semantic": None, "kg": None},
            }
        if src_type == "sem":
            val = _safe_float(item.get("score"), 0.0)
            candidates_map[key]["sem_score"] = max(candidates_map[key]["sem_score"], val)
            candidates_map[key]["sem_score_raw"] = max(candidates_map[key]["sem_score_raw"], val)
            candidates_map[key]["orig"]["semantic"] = item
            sem_scores.append(candidates_map[key]["sem_score"])
        else:
            val = max(
                _safe_float(item.get("kg_score_raw"), 0.0),
                _safe_float(item.get("kg_score"), 0.0),
                _safe_float(item.get("score"), 0.0),
            )
            candidates_map[key]["kg_score"] = max(candidates_map[key]["kg_score"], val)
            candidates_map[key]["kg_score_raw"] = max(candidates_map[key]["kg_score_raw"], val)
            candidates_map[key]["orig"]["kg"] = item
            candidates_map[key]["candidate_type"] = "KG"
            kg_scores.append(candidates_map[key]["kg_score"])
        candidates_map[key]["is_backfill"] = bool(candidates_map[key]["is_backfill"] or item.get("is_backfill", False))

        # merge entities
        ents = item.get("entities") or []
        merged = dedupe_list([e.lower().strip() for e in (candidates_map[key]["entities"] + ents) if e])
        candidates_map[key]["entities"] = merged

    # Add items
    for it in semantic_results or []:
        _add_candidate("sem", it)
    for it in kg_results or []:
        _add_candidate("kg", it)

    # Normalize sem and kg scores across candidates
    # sem_norm = _normalize_scores(sem_scores) if sem_scores else []
    # kg_norm = _normalize_scores(kg_scores) if kg_scores else []
    # Map original values to normalized quickly (we will consume them in order)
    # Build lists in the same order as sem_scores / kg_scores consumption above
    # But because candidates_map merged duplicates, safer to re-collect arrays keyed by candidate
    sem_values = [candidates_map[k]["sem_score"] for k in candidates_map]
    kg_values = [candidates_map[k]["kg_score"] for k in candidates_map]
    sem_norm_map = {k: v for k, v in zip(list(candidates_map.keys()), _normalize_scores(sem_values))}
    kg_norm_map = {k: v for k, v in zip(list(candidates_map.keys()), _normalize_scores(kg_values))}

    # Compute final score for each candidate
    results: List[Dict[str, Any]] = []
    now = now or datetime.now(timezone.utc)

    for key, item in candidates_map.items():
        sem_s = sem_norm_map.get(key, 0.0)
        kg_s = kg_norm_map.get(key, 0.0)
        kg_raw = _safe_float(item.get("kg_score_raw"), 0.0)
        ent_s = _entity_overlap_score(query_entities, item.get("entities", []))
        recency_s = _recency_boost(item.get("published_at"), now=now)
        cred_s = _safe_float(item.get("credibility"), 0.5)
        stmt_l = item["statement"].lower()
        if any(p in stmt_l for p in LOW_SIGNAL_PHRASES):
            continue

        claim_overlap = _claim_overlap_score(query_text, item["statement"])
        anchor_eval = evaluate_anchor_match(query_text, item["statement"])
        if not anchor_eval["anchor_ok"]:
            claim_overlap = min(claim_overlap, 0.20)
        stmt_lq = item["statement"].lower()
        claim_assertive = bool(
            re.search(
                r"\b(helps?|prevents?|reduces?|increases?|causes?|proves?|protects?)\b", query_text, re.IGNORECASE
            )
        )
        uncertainty_penalty = 0.0
        if claim_assertive and any(p in stmt_lq for p in UNCERTAINTY_PHRASES):
            uncertainty_penalty = 0.12

        final_score = (
            (w_sem * sem_s)
            + (w_kg * kg_s)
            + (w_entity * ent_s)
            + (w_claim_overlap * claim_overlap)
            + (w_recency * recency_s)
            + (w_cred * cred_s)
        )
        # Keep raw KG confidence relevant even when normalization compresses scores.
        if kg_raw > 0.0 and claim_overlap >= 0.08:
            final_score += 0.05 * min(1.0, kg_raw)
        # Reward KG-backed, entity-aligned evidence so KG can contribute in top-k.
        if kg_s > 0.0 and ent_s >= 0.34 and claim_overlap >= 0.10:
            final_score += 0.10
        if item.get("candidate_type") == "KG" and kg_raw >= 0.40 and anchor_eval.get("anchor_ok", True):
            final_score += 0.08
        is_backfill = bool(item.get("is_backfill", False))
        if is_backfill and not (ent_s >= 0.25 or kg_s >= 0.55 or kg_raw >= 0.55):
            final_score *= 0.85
        final_score -= uncertainty_penalty

        # small heuristic: if both sem and kg are zero but credibility high, ensure min floor
        if sem_s == 0.0 and kg_s == 0.0 and cred_s >= RANKING_MIN_CREDIBILITY_THRESHOLD:
            final_score = max(final_score, RANKING_MIN_SCORE_FLOOR)

        out = {
            "statement": item["statement"],
            "entities": item.get("entities", []),
            "source_url": item.get("source_url"),
            "published_at": item.get("published_at"),
            "sem_score": sem_s,
            "sem_score_raw": item.get("sem_score_raw", 0.0),
            "kg_score": kg_s,
            "kg_score_raw": kg_raw,
            "entity_overlap": ent_s,
            "claim_overlap": claim_overlap,
            "anchors_matched": int(anchor_eval.get("matched_groups", 0)),
            "anchors_required": int(anchor_eval.get("required_groups", 0)),
            "anchor_ok": bool(anchor_eval.get("anchor_ok", True)),
            "candidate_type": item.get("candidate_type", "VDB"),
            "is_backfill": is_backfill,
            "recency": recency_s,
            "credibility": cred_s,
            "final_score": max(0.0, min(final_score, 1.0)),
            "orig": item.get("orig", {}),
        }
        # Filter out low-overlap items unless strongly supported semantically
        # or by KG+entity alignment. This prevents KG-only off-topic drift.
        if (
            claim_overlap < RANKING_MIN_CLAIM_OVERLAP
            and sem_s < 0.75
            and not ((kg_s >= 0.55 or kg_raw >= 0.45) and ent_s >= 0.20 and claim_overlap >= 0.08)
        ):
            continue
        if sem_s < 0.20 and claim_overlap < 0.08 and ent_s < 0.20 and kg_raw < 0.40:
            continue
        results.append(out)

    # Sort descending by final_score, then by sem_score, then by credibility, then stable textual tie-breaker
    results.sort(key=lambda r: (-r["final_score"], -r["sem_score"], -r["credibility"], r["statement"]))

    kg_in_ranked = sum(1 for r in results if _safe_float(r.get("kg_score"), 0.0) > 0.0)
    kg_with_score = sum(
        1
        for r in kg_results or []
        if max(
            _safe_float(r.get("kg_score_raw"), 0.0),
            _safe_float(r.get("kg_score"), 0.0),
            _safe_float(r.get("score"), 0.0),
        )
        > 0.0
    )
    kg_max = max(
        [
            max(
                _safe_float(r.get("kg_score_raw"), 0.0),
                _safe_float(r.get("kg_score"), 0.0),
                _safe_float(r.get("score"), 0.0),
            )
            for r in (kg_results or [])
        ]
        or [0.0]
    )
    kg_in_top = sum(1 for r in (results[:5] if results else []) if _safe_float(r.get("kg_score"), 0.0) > 0.0)
    if kg_results and kg_in_ranked == 0 and results:
        logger.info(
            "[hybrid_rank] KG candidates present but none survived ranking filters (kg_raw=%d, ranked=%d)",
            len(kg_results),
            len(results),
        )

    logger.info(
        "[hybrid_rank] Ranked %d candidates. Top score: %s (kg_input=%d, kg_with_score=%d, "
        "max_kg_score=%.3f, kg_in_ranked=%d, kg_in_top=%d)",
        len(results),
        (results[0]["final_score"] if results else "N/A"),
        len(kg_results or []),
        kg_with_score,
        kg_max,
        kg_in_ranked,
        kg_in_top,
    )
    return results
