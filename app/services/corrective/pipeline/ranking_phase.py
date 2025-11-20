"""
Ranking Phase: Hybrid ranking of semantic and KG candidates with trust-based grading.
"""

from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.ranking.hybrid_ranker import hybrid_rank
from app.services.ranking.trust_ranker import TrustRanker

logger = get_logger(__name__)


async def rank_candidates(
    semantic_candidates: List[Dict[str, Any]],
    kg_candidates: List[Dict[str, Any]],
    query_entities: List[str],
    top_k: int,
    round_id: str,
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
    # Phase 1: Hybrid rank semantic and KG candidates
    ranked = hybrid_rank(semantic_candidates, kg_candidates, query_entities=query_entities)
    top_ranked = ranked[:top_k]

    # Phase 2: Enrich with trust grades
    graded_results = TrustRanker.enrich_ranked_results(top_ranked)

    score_str = top_ranked[0]["final_score"] if top_ranked else "N/A"
    logger.info(
        f"[RankingPhase:{round_id}] Ranked {len(ranked)} candidates (final score: {score_str}), "
        f"returned {len(graded_results)} top-k with trust grades"
    )

    return graded_results
