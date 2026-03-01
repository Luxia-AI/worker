from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config.trusted_domains import is_trusted_domain
from app.constants.config import (
    CREDIBILITY_AUTHORITY,
    CREDIBILITY_DEFAULT,
    RANKING_MIN_CLAIM_OVERLAP,
    RANKING_MIN_CREDIBILITY_THRESHOLD,
    RANKING_MIN_SCORE_FLOOR,
    RANKING_WEIGHTS,
    RECENCY_HALF_LIFE_DAYS,
)
from app.core.logger import get_logger
from app.services.common.list_ops import dedupe_list
from app.services.logic.evidence_strength import compute_negation_anchor_overlap
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


def _infer_claim_type(query_text: str) -> str:
    low = (query_text or "").lower()
    if re.search(r"\b(treat|treatment|cure|prevent|effective|efficacy|works?)\b", low):
        return "therapeutic"
    if re.search(r"\b(cause|causes|linked|association|risk|increase|decrease)\b", low):
        return "causal"
    if re.search(r"\b(mechanism|pathway|gene|protein|receptor|cellular)\b", low):
        return "mechanistic"
    return "general"


def _claim_type_weight_overrides(claim_type: str) -> Dict[str, float]:
    if claim_type == "therapeutic":
        return {
            "w_semantic": 0.35,
            "w_kg": 0.08,
            "w_entity": 0.20,
            "w_claim_overlap": 0.18,
            "w_recency": 0.08,
            "w_credibility": 0.08,
            "w_source_quality": 0.03,
        }
    if claim_type == "causal":
        return {
            "w_semantic": 0.28,
            "w_kg": 0.16,
            "w_entity": 0.19,
            "w_claim_overlap": 0.18,
            "w_recency": 0.06,
            "w_credibility": 0.09,
            "w_source_quality": 0.04,
        }
    if claim_type == "mechanistic":
        return {
            "w_semantic": 0.27,
            "w_kg": 0.20,
            "w_entity": 0.19,
            "w_claim_overlap": 0.16,
            "w_recency": 0.05,
            "w_credibility": 0.09,
            "w_source_quality": 0.04,
        }
    return {
        "w_semantic": RANKING_WEIGHTS["w_semantic"],
        "w_kg": RANKING_WEIGHTS["w_kg"],
        "w_entity": RANKING_WEIGHTS["w_entity"],
        "w_claim_overlap": RANKING_WEIGHTS["w_claim_overlap"],
        "w_recency": RANKING_WEIGHTS["w_recency"],
        "w_credibility": RANKING_WEIGHTS["w_credibility"],
        "w_source_quality": 0.03,
    }


def _source_quality_score(meta: Dict[str, Any]) -> float:
    source_url = str(meta.get("source_url") or meta.get("source") or "").lower()
    doc_type = str(meta.get("doc_type") or "").lower()
    statement = str(meta.get("statement") or "").lower()
    score = 0.5
    if doc_type in {"guideline"}:
        score = 0.95
    elif doc_type in {"journal"}:
        if any(k in statement for k in ("systematic review", "meta-analysis", "randomized")):
            score = 0.93
        else:
            score = 0.88
    elif doc_type in {"report"}:
        score = 0.80
    elif doc_type in {"web", "encyclopedia"}:
        score = 0.60
    elif doc_type in {"news"}:
        score = 0.45
    if "cochrane" in source_url:
        score = max(score, 0.95)
    if "pubmed.ncbi.nlm.nih.gov" in source_url or "pmc.ncbi.nlm.nih.gov" in source_url:
        score = max(score, 0.88)
    return max(0.0, min(1.0, score))


def _is_claim_mention_statement(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    patterns = (
        r"\bmyth\b",
        r"\bmisinformation\b",
        r"\bdisinformation\b",
        r"\bconspiracy\b",
        r"\bfalse claim\b",
        r"\bunfounded\b",
        r"\bhoax\b",
        r"\brumou?r\b",
        r"\bbeliev(?:e|ed|es|ing)\s+that\b",
        r"\bperceiv(?:e|ed|es|ing)\s+that\b",
        r"\bhesitan(?:cy|t)\b",
        r"\bacceptance\b",
        r"\bsurvey\b",
    )
    return any(re.search(p, low) for p in patterns)


def _claim_is_belief_or_survey(claim_text: str) -> bool:
    if not claim_text:
        return False
    low = claim_text.lower()
    return bool(
        re.search(
            r"\b(believe|belief|believed|think|thought|perceive|perceived|concern|hesitancy|acceptance|survey)\b",
            low,
        )
    )


def _object_tokens_for_query(text: str) -> set[str]:
    if not text:
        return set()
    m = re.search(r"\b(?:against|for|on)\s+(.+)$", text.lower())
    if not m:
        return set()
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
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
    return {w for w in re.findall(r"\b[\w']+\b", m.group(1)) if w not in stop and len(w) > 2}


def _statement_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\b[\w']+\b", (text or "").lower()) if len(w) > 2}


def _action_markers(text: str) -> set[str]:
    low = (text or "").lower()
    markers: set[str] = set()
    if re.search(
        (
            r"\b(improv(?:e|es|ed|ing)|enhanc(?:e|es|ed|ing)|"
            r"benefit(?:s|ed|ing)?|help(?:s|ed|ing)?|"
            r"reliev(?:e|es|ed|ing)|facilitat(?:e|es|ed|ing))\b"
        ),
        low,
    ):
        markers.add("improve")
    if re.search(r"\b(reduc(?:e|es|ed|ing)|lower(?:s|ed|ing)?|decreas(?:e|es|ed|ing))\b", low):
        markers.add("reduce")
    if re.search(r"\b(prevent(?:s|ed|ing)?|protect(?:s|ed|ing)?)\b", low):
        markers.add("prevent")
    if re.search(r"\b(caus(?:e|es|ed|ing)|trigger(?:s|ed|ing)?|increas(?:e|es|ed|ing))\b", low):
        markers.add("cause")
    if re.search(r"\b(treat(?:s|ed|ing)?|cure(?:s|d|ing)?|manag(?:e|es|ed|ing))\b", low):
        markers.add("treat")
    if re.search(r"\b(diagnos(?:e|es|ed|ing)|detect(?:s|ed|ing)?|determin(?:e|es|ed|ing)|test(?:s|ed|ing)?)\b", low):
        markers.add("diagnose")
    return markers


def _has_object_refutation_signal(statement: str) -> bool:
    low = (statement or "").lower()
    patterns = (
        r"\bonly\s+(?:for|against)\b",
        r"\bdo(?:es)?\s+not\s+work\s+(?:for|against|on)\b",
        r"\bnot\s+effective\s+(?:for|against)\b",
        r"\bineffective\s+(?:for|against)\b",
        r"\bviral\b",
    )
    return any(re.search(p, low) for p in patterns)


def _claim_focus_tokens(text: str) -> set[str]:
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
        "this",
        "that",
        "these",
        "those",
    }
    tokens = {w for w in re.findall(r"\b[\w']+\b", (text or "").lower()) if w not in stop and len(w) > 3}
    return tokens


def _relation_focus_tokens(text: str) -> tuple[set[str], set[str]]:
    """
    Extract coarse subject/object token sets for directional claims, e.g.:
    "smoking increases risk of lung cancer" -> subject={smoking}, object={lung,cancer}
    """
    if not text:
        return set(), set()
    low = text.lower().strip()
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
        "risk",
    }
    m = re.search(
        r"^(?P<subj>.+?)\b(?:increase|increases|increased|cause|causes|caused|"
        r"reduce|reduces|reduced|prevent|prevents|prevented|has|have)\b(?P<obj>.+)$",
        low,
    )
    if not m:
        return set(), set()
    subj = {w for w in re.findall(r"\b[\w']+\b", m.group("subj")) if len(w) > 2 and w not in stop}
    obj = {w for w in re.findall(r"\b[\w']+\b", m.group("obj")) if len(w) > 2 and w not in stop}
    return subj, obj


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

    # Canonical trusted-domain heuristic.
    source_url = (meta.get("source_url") or meta.get("source") or "").strip()
    if source_url and is_trusted_domain(source_url):
        return CREDIBILITY_AUTHORITY

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
    claim_type = _infer_claim_type(query_text)
    claim_weight_defaults = _claim_type_weight_overrides(claim_type)
    # weights defaults
    w_sem = float(weights.get("w_semantic", claim_weight_defaults["w_semantic"]))
    w_kg = float(weights.get("w_kg", claim_weight_defaults["w_kg"]))
    w_entity = float(weights.get("w_entity", claim_weight_defaults["w_entity"]))
    w_claim_overlap = float(weights.get("w_claim_overlap", claim_weight_defaults["w_claim_overlap"]))
    w_recency = float(weights.get("w_recency", claim_weight_defaults["w_recency"]))
    w_cred = float(weights.get("w_credibility", claim_weight_defaults["w_credibility"]))
    w_source_quality = float(weights.get("w_source_quality", claim_weight_defaults["w_source_quality"]))

    # Determine if we have enough query context for text-based filtering/penalties.
    # When query_text is absent or trivial, claim_overlap/anchor/focus signals are
    # undefined and must NOT be used to penalise or filter candidates.
    has_query_context = bool(query_text and len(query_text.strip()) > 3)

    _debug = os.getenv("HYBRID_RANK_DEBUG", "").lower() in ("1", "true", "yes")

    # Build unified candidate list keyed by (statement, source_url) to merge duplicates
    candidates_map: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    # Collect raw scores for normalization
    sem_scores = []
    kg_scores = []

    # Normalize input fields
    def _add_candidate(src_type: str, item: Dict[str, Any]) -> None:
        stmt = (item.get("statement") or "").strip()
        source_url = item.get("source_url") or item.get("source")
        # canonical key: (normalised statement, source_url)
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

        # merge entities (dedup, preserve both sources)
        ents = item.get("entities") or []
        merged = dedupe_list([e.lower().strip() for e in (candidates_map[key]["entities"] + ents) if e])
        candidates_map[key]["entities"] = merged

    # Add items
    for it in semantic_results or []:
        _add_candidate("sem", it)
    for it in kg_results or []:
        _add_candidate("kg", it)

    # Normalize sem and kg scores across candidates
    sem_values = [candidates_map[k]["sem_score"] for k in candidates_map]
    kg_values = [candidates_map[k]["kg_score"] for k in candidates_map]
    sem_norm_map = {k: v for k, v in zip(list(candidates_map.keys()), _normalize_scores(sem_values))}
    kg_norm_map = {k: v for k, v in zip(list(candidates_map.keys()), _normalize_scores(kg_values))}

    # ------------------------------------------------------------------
    # Score every candidate.  We collect ALL scored items first, then
    # apply admission filters in a second pass so that KG minimum
    # retention can rescue dropped KG candidates.
    # ------------------------------------------------------------------
    all_scored: List[Dict[str, Any]] = []
    now = now or datetime.now(timezone.utc)
    query_object_tokens = _object_tokens_for_query(query_text)
    claim_focus_tokens = _claim_focus_tokens(query_text)
    subject_focus_tokens, object_focus_tokens = _relation_focus_tokens(query_text)
    claim_actions = _action_markers(query_text)
    belief_claim = _claim_is_belief_or_survey(query_text)

    for key, item in candidates_map.items():
        sem_s = sem_norm_map.get(key, 0.0)
        kg_s = kg_norm_map.get(key, 0.0)
        kg_raw = _safe_float(item.get("kg_score_raw"), 0.0)
        ent_s = _entity_overlap_score(query_entities, item.get("entities", []))
        recency_s = _recency_boost(item.get("published_at"), now=now)
        cred_s = _safe_float(item.get("credibility"), 0.5)
        source_quality_s = _source_quality_score(item)
        stmt_l = item["statement"].lower()

        # Hard filter: low-signal boilerplate pages
        if any(p in stmt_l for p in LOW_SIGNAL_PHRASES):
            continue

        claim_overlap = _claim_overlap_score(query_text, item["statement"])
        anchor_eval = evaluate_anchor_match(query_text, item["statement"])
        anchor_match_score = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
        negation_anchor_overlap = float(compute_negation_anchor_overlap(query_text, item["statement"]) or 0.0)
        if has_query_context and not anchor_eval["anchor_ok"]:
            claim_overlap = min(claim_overlap, 0.20)
        stmt_lq = item["statement"].lower()
        claim_assertive = (
            bool(
                re.search(
                    r"\b(helps?|prevents?|reduces?|increases?|causes?|proves?|protects?)\b", query_text, re.IGNORECASE
                )
            )
            if has_query_context
            else False
        )
        uncertainty_penalty = 0.0
        if claim_assertive and any(p in stmt_lq for p in UNCERTAINTY_PHRASES):
            uncertainty_penalty = 0.12

        # ---- Base weighted sum ----
        final_score = (
            (w_sem * sem_s)
            + (w_kg * kg_s)
            + (w_entity * ent_s)
            + (w_claim_overlap * claim_overlap)
            + (w_recency * recency_s)
            + (w_cred * cred_s)
            + (w_source_quality * source_quality_s)
        )

        # ---- Authority credibility bonus (general, not claim/entity specific) ----
        # Health claim verification gives significant priority to authoritative sources.
        credibility_bonus = 0.0
        if cred_s >= 0.90:
            credibility_bonus = 0.30
        elif cred_s >= 0.75:
            credibility_bonus = 0.10
        final_score += credibility_bonus

        # ---- Additive KG bonuses ----
        neg_anchor_weight = float(os.getenv("NEGATION_ANCHOR_BOOST_WEIGHT", "0.07"))
        if negation_anchor_overlap > 0.0:
            final_score += neg_anchor_weight * max(0.0, min(1.0, negation_anchor_overlap))
        # Keep raw KG confidence relevant; relax claim_overlap gate when no query context.
        if kg_raw > 0.0 and (claim_overlap >= 0.08 or not has_query_context):
            final_score += 0.05 * min(1.0, kg_raw)
        # Reward KG-backed, entity-aligned evidence so KG can contribute in top-k.
        if kg_s > 0.0 and ent_s >= 0.34 and (claim_overlap >= 0.10 or not has_query_context):
            final_score += 0.10
        if item.get("candidate_type") == "KG" and kg_raw >= 0.40 and anchor_eval.get("anchor_ok", True):
            final_score += 0.08

        candidate_type = str(item.get("candidate_type", "VDB"))
        is_backfill = bool(item.get("is_backfill", False))

        # Compute text-alignment features (needed for output even when no penalties)
        stmt_tokens = _statement_tokens(item["statement"])
        stmt_actions = _action_markers(item["statement"])
        action_overlap_ok = not claim_actions or bool(claim_actions & stmt_actions)
        object_overlap = len(query_object_tokens & stmt_tokens) if query_object_tokens else 0
        focus_overlap = len(claim_focus_tokens & stmt_tokens) if claim_focus_tokens else 0
        subject_overlap = len(subject_focus_tokens & stmt_tokens) if subject_focus_tokens else 0
        object_relation_overlap = len(object_focus_tokens & stmt_tokens) if object_focus_tokens else 0

        # ---- Alignment penalties: ONLY when query context exists ----
        # Uses bounded additive penalty instead of multiplicative chaining to
        # prevent score collapse.  Each misalignment contributes a penalty
        # strength; the combined penalty is capped.
        if has_query_context:
            penalties: List[float] = []

            if anchor_match_score < 0.20:
                penalties.append(0.50 if candidate_type == "KG" else 0.25)

            if is_backfill and not (ent_s >= 0.25 or kg_s >= 0.55 or kg_raw >= 0.55):
                penalties.append(0.12)

            if claim_actions and not action_overlap_ok and sem_s < 0.95 and kg_raw < 0.95:
                penalties.append(0.20)

            if query_object_tokens and object_overlap == 0 and not _has_object_refutation_signal(item["statement"]):
                penalties.append(0.30)

            if claim_focus_tokens and focus_overlap <= 1 and sem_s < 0.85 and kg_raw < 0.85:
                penalties.append(0.18)

            if subject_focus_tokens and subject_overlap == 0 and sem_s < 0.90 and kg_raw < 0.90:
                penalties.append(0.25)

            if object_focus_tokens and object_relation_overlap == 0 and sem_s < 0.90 and kg_raw < 0.90:
                penalties.append(0.20)

            if subject_focus_tokens and object_focus_tokens and subject_overlap == 0 and object_relation_overlap == 0:
                penalties.append(0.40)

            if not belief_claim and _is_claim_mention_statement(item["statement"]):
                penalties.append(0.25)

            # Bounded combined penalty: worst penalty dominates, others add
            # with diminishing returns.  Caps at 65% total reduction.
            if penalties:
                penalties.sort(reverse=True)
                combined = penalties[0]
                for p in penalties[1:]:
                    combined += p * 0.25
                combined = min(combined, 0.65)
                final_score *= 1.0 - combined

        final_score -= uncertainty_penalty

        # Min floor when both sem and kg are zero but credibility high
        if sem_s == 0.0 and kg_s == 0.0 and cred_s >= RANKING_MIN_CREDIBILITY_THRESHOLD:
            final_score = max(final_score, RANKING_MIN_SCORE_FLOOR)

        # ---- Support/contradict stance scoring ----
        predicate_match_score = 0.0
        if claim_actions:
            predicate_match_score = 1.0 if action_overlap_ok else 0.0
            if claim_overlap >= 0.20:
                predicate_match_score = max(predicate_match_score, min(1.0, claim_overlap))
        else:
            predicate_match_score = min(1.0, max(claim_overlap, anchor_match_score))

        object_align = 1.0 if object_relation_overlap > 0 else (0.4 if object_overlap > 0 else 0.0)
        scope_align = max(0.0, min(1.0, (0.55 * claim_overlap) + (0.45 * anchor_match_score)))
        support_score = max(
            0.0,
            min(
                1.0,
                (0.30 * predicate_match_score)
                + (0.20 * object_align)
                + (0.15 * source_quality_s)
                + (0.15 * scope_align)
                + (0.20 * max(kg_s, min(1.0, kg_raw))),
            ),
        )
        explicit_refute = 1.0 if _has_object_refutation_signal(item["statement"]) else 0.0
        predicate_conflict = 1.0 if (claim_actions and not action_overlap_ok) else 0.0
        kg_refute_path = max(kg_s, min(1.0, kg_raw)) if (explicit_refute > 0.0 or predicate_conflict > 0.0) else 0.0
        contradict_score = max(
            0.0,
            min(
                1.0,
                (0.38 * max(negation_anchor_overlap, explicit_refute))
                + (0.22 * predicate_conflict)
                + (0.18 * max(0.0, 1.0 - predicate_match_score))
                + (0.22 * kg_refute_path),
            ),
        )
        final_rank_priority = max(support_score, contradict_score)

        # Soft blend with stance priority (0.95 base / 0.05 priority) to
        # avoid compressing the base score.  Priority acts primarily as a
        # tiebreaker rather than a dominating term.
        final_score = (0.95 * max(0.0, min(1.0, final_score))) + (0.05 * final_rank_priority)

        recency_score = recency_s  # alias for output clarity

        out = {
            "statement": item["statement"],
            "entities": item.get("entities", []),
            "source_url": item.get("source_url"),
            "published_at": item.get("published_at"),
            "sem_score": w_sem * sem_s,
            "sem_score_raw": item.get("sem_score_raw", 0.0),
            "kg_score": kg_s,
            "kg_score_raw": kg_raw,
            "entity_overlap": ent_s,
            "claim_overlap": claim_overlap,
            "source_quality": source_quality_s,
            "claim_type_hint": claim_type,
            "anchors_matched": int(anchor_eval.get("matched_groups", 0)),
            "anchors_required": int(anchor_eval.get("required_groups", 0)),
            "anchor_match_score": anchor_match_score,
            "predicate_match_score": predicate_match_score,
            "negation_anchor_overlap": negation_anchor_overlap,
            "anchor_ok": bool(anchor_eval.get("anchor_ok", True)),
            "candidate_type": candidate_type,
            "is_backfill": is_backfill,
            "subject_overlap": subject_overlap,
            "object_relation_overlap": object_relation_overlap,
            "focus_overlap": focus_overlap,
            "action_overlap_ok": bool(action_overlap_ok),
            "recency": recency_s,
            "recency_score": recency_score,
            "credibility": cred_s,
            "credibility_bonus": credibility_bonus,
            "support_score": support_score,
            "contradict_score": contradict_score,
            "final_rank_priority": final_rank_priority,
            "final_score": max(0.0, min(final_score, 1.0)),
            "orig": item.get("orig", {}),
        }

        if _debug:
            logger.debug(
                "[hybrid_rank][debug] candidate=%s src=%s sem_s=%.3f kg_s=%.3f kg_raw=%.3f "
                "ent_s=%.3f cred=%.3f cred_bonus=%.2f claim_overlap=%.3f "
                "final=%.4f has_ctx=%s",
                item["statement"][:60],
                candidate_type,
                sem_s,
                kg_s,
                kg_raw,
                ent_s,
                cred_s,
                credibility_bonus,
                claim_overlap,
                out["final_score"],
                has_query_context,
            )

        all_scored.append(out)

    # ------------------------------------------------------------------
    # Admission filters (separate pass so KG retention can rescue).
    # Gate text-dependent filters on has_query_context.
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    for out in all_scored:
        # Recover normalised sem_s for filter thresholds
        # (sem_score output is now the weighted component; use raw for filters)
        _sem_norm = out["sem_score"] / w_sem if w_sem > 0 and out["sem_score"] > 0 else 0.0
        _kg_raw = out["kg_score_raw"]
        _ent_s = out["entity_overlap"]
        _claim_ov = out["claim_overlap"]
        _kg_s = out["kg_score"]
        _cred = out["credibility"]

        if has_query_context:
            # Filter out low-overlap items unless strongly supported semantically
            # or by KG+entity alignment.
            if (
                _claim_ov < RANKING_MIN_CLAIM_OVERLAP
                and _sem_norm < 0.75
                and not ((_kg_s >= 0.55 or _kg_raw >= 0.45) and _ent_s >= 0.20 and _claim_ov >= 0.08)
            ):
                continue
            if (
                query_object_tokens
                and out.get("subject_overlap", 0) == 0
                and out.get("object_relation_overlap", 0) == 0
                and _claim_ov < 0.40
                and _sem_norm < 0.90
                and _kg_raw < 0.90
            ):
                if not _has_object_refutation_signal(out["statement"]):
                    continue
            if (
                claim_focus_tokens
                and out.get("focus_overlap", 0) <= 1
                and _claim_ov < 0.35
                and _sem_norm < 0.90
                and _kg_raw < 0.90
            ):
                continue
            if subject_focus_tokens and out.get("subject_overlap", 0) == 0 and _sem_norm < 0.95 and _kg_raw < 0.95:
                continue
            if (
                object_focus_tokens
                and out.get("object_relation_overlap", 0) == 0
                and _sem_norm < 0.95
                and _kg_raw < 0.95
            ):
                continue
            if not belief_claim and _is_claim_mention_statement(out["statement"]) and _claim_ov < 0.60:
                continue
            if (
                claim_actions
                and not out.get("action_overlap_ok", True)
                and object_focus_tokens
                and out.get("object_relation_overlap", 0) == 0
                and _claim_ov < 0.75
                and _kg_raw < 0.95
            ):
                continue
            if (
                claim_actions
                and not out.get("action_overlap_ok", True)
                and _claim_ov < 0.45
                and _sem_norm < 0.92
                and _kg_raw < 0.92
            ):
                continue
            # Drop truly zero-signal items
            if _sem_norm < 0.20 and _claim_ov < 0.08 and _ent_s < 0.20 and _kg_raw < 0.40:
                continue
        else:
            # Without query context, only drop absolute noise (no signal at all
            # and very low credibility).
            if _sem_norm == 0.0 and _kg_raw == 0.0 and _ent_s == 0.0 and _cred < 0.15 and out["final_score"] < 0.05:
                continue

        results.append(out)

    # ------------------------------------------------------------------
    # KG minimum retention policy: if KG candidates existed in input but
    # none survived filters, re-admit the best KG candidate with
    # adequate score.  This is a general diversity rule, not entity-specific.
    # ------------------------------------------------------------------
    kg_in_ranked = sum(
        1 for r in results if _safe_float(r.get("kg_score"), 0.0) > 0.0 or r.get("candidate_type") == "KG"
    )
    if kg_results and kg_in_ranked == 0:
        kg_dropped = [
            s
            for s in all_scored
            if (s.get("candidate_type") == "KG" or s.get("kg_score_raw", 0) > 0) and s not in results
        ]
        if kg_dropped:
            best_kg = max(kg_dropped, key=lambda x: x.get("final_score", 0))
            if best_kg.get("kg_score_raw", 0) >= 0.30 or best_kg.get("final_score", 0) >= 0.10:
                results.append(best_kg)
                if _debug:
                    logger.debug(
                        "[hybrid_rank][debug] KG retention rescued: %s (final=%.4f)",
                        best_kg["statement"][:60],
                        best_kg["final_score"],
                    )

    # ------------------------------------------------------------------
    # Deterministic sort: final_score desc, then credibility desc, then
    # support_score desc, then recency desc, then statement asc (stable
    # alphabetical tie-break).
    # ------------------------------------------------------------------
    results.sort(
        key=lambda r: (
            -r["final_score"],
            -r.get("credibility", 0),
            -r.get("support_score", 0),
            -r.get("recency", 0),
            r["statement"],
        )
    )

    # ------------------------------------------------------------------
    # Diagnostics logging
    # ------------------------------------------------------------------
    kg_in_ranked = sum(
        1 for r in results if _safe_float(r.get("kg_score"), 0.0) > 0.0 or r.get("candidate_type") == "KG"
    )
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
    vdb_in_ranked = sum(1 for r in results if r.get("candidate_type") != "KG")
    kg_in_top = sum(
        1
        for r in (results[:5] if results else [])
        if _safe_float(r.get("kg_score"), 0.0) > 0.0 or r.get("candidate_type") == "KG"
    )
    if kg_results and kg_in_ranked == 0 and results:
        logger.info(
            "[hybrid_rank] KG candidates present but none survived ranking filters (kg_raw=%d, ranked=%d)",
            len(kg_results),
            len(results),
        )

    logger.info(
        "[hybrid_rank] Ranked %d candidates (vdb=%d, kg=%d). Top score: %s "
        "(kg_input=%d, kg_with_score=%d, max_kg_score=%.3f, kg_in_ranked=%d, kg_in_top=%d)",
        len(results),
        vdb_in_ranked,
        kg_in_ranked,
        (f"{results[0]['final_score']:.4f}" if results else "N/A"),
        len(kg_results or []),
        kg_with_score,
        kg_max,
        kg_in_ranked,
        kg_in_top,
    )
    return results
