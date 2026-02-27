"""
Ranking Phase: Hybrid ranking of semantic and KG candidates with trust-based grading.
"""

import re
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger, log_value_payload
from app.services.logging.log_manager import LogManager
from app.services.ranking.contradiction_scorer import ContradictionScorer
from app.services.ranking.hybrid_ranker import hybrid_rank
from app.services.ranking.trust_ranker import DummyStanceClassifier, TrustRanker
from app.services.retrieval.evidence_gate import filter_candidates_for_count_claim

logger = get_logger(__name__)


def _tokenize(text: str) -> set[str]:
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
    return {w for w in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", (text or "").lower()) if w not in stop}


def _entity_anchor_tokens(entities: List[str]) -> set[str]:
    toks: set[str] = set()
    for e in entities or []:
        toks |= _tokenize(e)
    return toks


def _topic_aligned_candidate(candidate: Dict[str, Any], claim_text: str, claim_entities: List[str]) -> bool:
    statement = str(candidate.get("statement") or "")
    if not statement:
        return False
    stmt_toks = _tokenize(statement)
    if not stmt_toks:
        return False
    claim_toks = _tokenize(claim_text)
    entity_toks = _entity_anchor_tokens(claim_entities)

    predicate_match = float(candidate.get("predicate_match_score", 0.0) or 0.0)
    object_overlap = int(candidate.get("object_relation_overlap", 0) or 0)
    claim_overlap = float(candidate.get("claim_overlap", 0.0) or 0.0)
    anchor_overlap = float(candidate.get("anchor_match_score", 0.0) or 0.0)

    entity_hit = bool(entity_toks and (stmt_toks & entity_toks))
    lexical_hit = bool(claim_toks and (stmt_toks & claim_toks))
    if entity_hit and (claim_overlap >= 0.10 or anchor_overlap >= 0.15):
        return True
    if lexical_hit and (predicate_match >= 0.20 or object_overlap >= 1):
        return True
    return False


def _contains_must_have_entity(candidate: Dict[str, Any], must_have_aliases: List[str]) -> bool:
    if not must_have_aliases:
        return True
    tokens = _tokenize(str(candidate.get("statement") or ""))
    if not tokens:
        return False
    for alias in must_have_aliases:
        alias_tokens = _tokenize(str(alias or ""))
        if not alias_tokens:
            continue
        overlap = len(tokens & alias_tokens)
        if len(alias_tokens) == 1 and overlap >= 1:
            return True
        if len(alias_tokens) > 1 and overlap >= max(1, len(alias_tokens) - 1):
            return True
    return False


def _alias_overlap_score(candidate: Dict[str, Any], must_have_aliases: List[str]) -> float:
    """Soft overlap score used when strict alias matching fails for all top items."""
    if not must_have_aliases:
        return 0.0
    tokens = _tokenize(str(candidate.get("statement") or ""))
    if not tokens:
        return 0.0
    best = 0.0
    for alias in must_have_aliases:
        alias_tokens = _tokenize(str(alias or ""))
        if not alias_tokens:
            continue
        overlap = len(tokens & alias_tokens)
        ratio = overlap / max(1, len(alias_tokens))
        if ratio > best:
            best = ratio
    return best


async def rank_candidates(
    semantic_candidates: List[Dict[str, Any]],
    kg_candidates: List[Dict[str, Any]],
    query_entities: List[str],
    query_text: str,
    top_k: int,
    round_id: str,
    log_manager: Optional[LogManager] = None,
    must_have_entities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid ranking on semantic and KG candidates, then assign trust grades.

    Args:
        semantic_candidates: List of semantic results from VDB with keys:
            - score: float (cosine similarity [0, 1])
            - statement: str
            - entities: List[str]
            - source_url: str
            - published_at: Optional[str]
            - credibility: Optional[float]
        kg_candidates: List of KG results from Neo4j with keys:
            - score: float (path quality [0, 1])
            - statement: str
            - entities: List[str]
            - source_url: Optional[str]
            - credibility: float [0, 1]
        query_entities: Entities extracted from original post
        query_text: Full claim text (used for overlap-aware ranking)
        top_k: Number of top results to return
        round_id: Round identifier for logging

    Returns:
        List of ranked candidates with trust grades, sorted by final_score (descending).
        Each item includes:
            - All hybrid_rank fields (statement, entities, final_score, etc.)
            - grade: str (A+, A, B, C, D, F)
            - semantic_confidence: str (HIGH, GOOD, FAIR, LOW, NONE)
            - grade_rationale: str (explanation of grade)
    """
    # Gate evidence for count-claims (prevents unrelated results)
    semantic_candidates, kg_candidates = filter_candidates_for_count_claim(
        semantic_candidates,
        kg_candidates,
        query_text,
        query_entities,
    )

    # Phase 1: Hybrid rank semantic and KG candidates
    ranked = hybrid_rank(
        semantic_candidates,
        kg_candidates,
        query_entities=query_entities,
        query_text=query_text,
    )
    filtered_for_trust: List[Dict[str, Any]] = []
    for item in ranked:
        topic_aligned = _topic_aligned_candidate(item, query_text, query_entities)
        item["topic_aligned"] = bool(topic_aligned)
        item["admissible_for_trust"] = bool(topic_aligned)
        if topic_aligned:
            filtered_for_trust.append(item)
    if filtered_for_trust:
        ranked = filtered_for_trust
    stance_classifier = DummyStanceClassifier()
    scorer = ContradictionScorer(semantic_min=0.35)
    for item in ranked:
        statement = str(item.get("statement") or item.get("text") or "")
        stance = stance_classifier.classify_stance(query_text, statement) if statement else "neutral"
        item["stance"] = stance
        if stance == "contradicts":
            raw_score = float(item.get("final_score", 0.0) or 0.0)
            penalized = max(0.0, raw_score * 0.60)
            item["raw_final_score"] = raw_score
            item["final_score"] = penalized
            item["contradiction_penalty"] = round(raw_score - penalized, 4)
    contradicting_count = scorer.score(ranked).contra_count

    ranked.sort(
        key=lambda r: (
            -float(r.get("final_score", 0.0) or 0.0),
            -float(r.get("sem_score", 0.0) or 0.0),
            -float(r.get("credibility", 0.0) or 0.0),
            str(r.get("statement", "")),
        )
    )

    # Keep contradiction evidence out of supporting top-k unless no non-contradicting
    # candidates exist (degraded mode).
    non_contradicting = [r for r in ranked if (r.get("stance") or "neutral") != "contradicts"]
    contradicting = [r for r in ranked if (r.get("stance") or "neutral") == "contradicts"]
    top_ranked = non_contradicting[:top_k] if non_contradicting else contradicting[:top_k]
    must_have_aliases = [x for x in (must_have_entities or []) if str(x).strip()]
    if must_have_aliases and top_ranked:
        with_core = [r for r in top_ranked if _contains_must_have_entity(r, must_have_aliases)]
        if with_core:
            without_core = [r for r in top_ranked if r not in with_core]
            top_ranked = (with_core + without_core)[:top_k]
        else:
            logger.warning(
                "[RankingPhase:%s] Soft gate fallback: top evidence missing strict core aliases=%s; "
                "keeping best overlap candidates instead of emptying ranking",
                round_id,
                must_have_aliases,
            )
            scored = [
                (
                    _alias_overlap_score(r, must_have_aliases),
                    float(r.get("final_score", 0.0) or 0.0),
                    r,
                )
                for r in top_ranked
            ]
            scored.sort(key=lambda x: (-x[0], -x[1]))
            top_ranked = [item[2] for item in scored[:top_k]]

    # Phase 2: Enrich with trust grades
    graded_results = TrustRanker.enrich_ranked_results(top_ranked)
    kg_in_top = sum(1 for r in graded_results if float(r.get("kg_score", 0.0) or 0.0) > 0.0)
    kg_in_ranked = sum(1 for r in ranked if float(r.get("kg_score", 0.0) or 0.0) > 0.0)
    sem_in_top = sum(1 for r in graded_results if float(r.get("sem_score", 0.0) or 0.0) > 0.0)

    top_score = float(top_ranked[0]["final_score"]) if top_ranked else 0.0
    avg_score = sum(float(r.get("final_score", 0.0) or 0.0) for r in ranked) / len(ranked) if ranked else 0.0
    top_k_selected = [
        {
            "statement": r.get("statement", ""),
            "source": r.get("source_url", ""),
            "semantic": round(float(r.get("sem_score", 0.0) or 0.0), 4),
            "kg": round(float(r.get("kg_score", 0.0) or 0.0), 4),
            "final": round(float(r.get("final_score", 0.0) or 0.0), 4),
            "stance": r.get("stance", "neutral"),
            "credibility": round(float(r.get("credibility", 0.0) or 0.0), 4),
        }
        for r in graded_results
    ]
    vdb_signal_sum_top5 = sum(float(r.get("sem_score", 0.0) or 0.0) for r in graded_results[:5])
    kg_signal_sum_top5 = sum(float(r.get("kg_score", 0.0) or 0.0) for r in graded_results[:5])
    log_value_payload(
        logger,
        "ranking",
        {
            "round_id": round_id,
            "top_k_selected": top_k_selected,
            "top_score": round(top_score, 4),
            "avg_score": round(avg_score, 4),
            "vdb_signal_sum_top5": round(vdb_signal_sum_top5, 4),
            "kg_signal_sum_top5": round(kg_signal_sum_top5, 4),
            "kg_in_ranked": kg_in_ranked,
            "kg_in_top": kg_in_top,
            "sem_in_top": sem_in_top,
            "contradicting_in_ranked": contradicting_count,
        },
    )

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"[PhaseOutput] ranking total_ranked={len(ranked)} top_k={len(graded_results)}",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={
                "total_ranked": len(ranked),
                "top_k_returned": len(graded_results),
                "top_score": top_score,
                "top_grade": top_ranked[0].get("grade") if top_ranked else "N/A",
                "avg_score": avg_score,
                "top_k_selected": top_k_selected,
                "vdb_signal_sum_top5": vdb_signal_sum_top5,
                "kg_signal_sum_top5": kg_signal_sum_top5,
                "kg_in_ranked": kg_in_ranked,
                "kg_in_top": kg_in_top,
                "sem_in_top": sem_in_top,
                "contradicting_in_ranked": contradicting_count,
            },
        )

    return graded_results
