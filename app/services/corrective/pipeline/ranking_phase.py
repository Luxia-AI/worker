"""
Ranking Phase: Hybrid ranking of semantic and KG candidates with trust-based grading.
"""

from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.services.logging.log_manager import LogManager
from app.services.ranking.hybrid_ranker import hybrid_rank
from app.services.ranking.trust_ranker import DummyStanceClassifier, TrustRanker
from app.services.retrieval.evidence_gate import filter_candidates_for_count_claim

logger = get_logger(__name__)


async def rank_candidates(
    semantic_candidates: List[Dict[str, Any]],
    kg_candidates: List[Dict[str, Any]],
    query_entities: List[str],
    query_text: str,
    top_k: int,
    round_id: str,
    log_manager: Optional[LogManager] = None,
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
    stance_classifier = DummyStanceClassifier()
    contradicting_count = 0
    for item in ranked:
        statement = str(item.get("statement") or item.get("text") or "")
        stance = stance_classifier.classify_stance(query_text, statement) if statement else "neutral"
        item["stance"] = stance
        if stance == "contradicts":
            contradicting_count += 1
            raw_score = float(item.get("final_score", 0.0) or 0.0)
            penalized = max(0.0, raw_score * 0.60)
            item["raw_final_score"] = raw_score
            item["final_score"] = penalized
            item["contradiction_penalty"] = round(raw_score - penalized, 4)

    ranked.sort(
        key=lambda r: (
            -float(r.get("final_score", 0.0) or 0.0),
            -float(r.get("sem_score", 0.0) or 0.0),
            -float(r.get("credibility", 0.0) or 0.0),
            str(r.get("statement", "")),
        )
    )

    # Keep contradiction evidence available, but prioritize non-contradicting items in top-k.
    non_contradicting = [r for r in ranked if (r.get("stance") or "neutral") != "contradicts"]
    contradicting = [r for r in ranked if (r.get("stance") or "neutral") == "contradicts"]
    top_ranked = (non_contradicting + contradicting)[:top_k]

    # Phase 2: Enrich with trust grades
    graded_results = TrustRanker.enrich_ranked_results(top_ranked)
    kg_in_top = sum(1 for r in graded_results if float(r.get("kg_score", 0.0) or 0.0) > 0.0)
    kg_in_ranked = sum(1 for r in ranked if float(r.get("kg_score", 0.0) or 0.0) > 0.0)
    sem_in_top = sum(1 for r in graded_results if float(r.get("sem_score", 0.0) or 0.0) > 0.0)

    score_str = top_ranked[0]["final_score"] if top_ranked else "N/A"
    logger.info(
        f"[RankingPhase:{round_id}] Ranked {len(ranked)} candidates (final score: {score_str}), "
        f"returned {len(graded_results)} top-k with trust grades "
        f"(kg_in_ranked={kg_in_ranked}, kg_in_top={kg_in_top}, sem_in_top={sem_in_top}, "
        f"contradicting_in_ranked={contradicting_count})"
    )

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"Ranking completed: {len(ranked)} candidates, top score: {score_str}",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={
                "total_ranked": len(ranked),
                "top_k_returned": len(graded_results),
                "top_score": top_ranked[0]["final_score"] if top_ranked else 0.0,
                "top_grade": top_ranked[0].get("grade") if top_ranked else "N/A",
                "kg_in_ranked": kg_in_ranked,
                "kg_in_top": kg_in_top,
                "sem_in_top": sem_in_top,
                "contradicting_in_ranked": contradicting_count,
            },
        )

    return graded_results
