"""
Ranking Phase: Hybrid ranking of semantic and KG candidates.
"""

from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.ranking.hybrid_ranker import hybrid_rank

logger = get_logger(__name__)


async def rank_candidates(
    semantic_candidates: List[Dict[str, Any]],
    kg_candidates: List[Dict[str, Any]],
    query_entities: List[str],
    top_k: int,
    round_id: str,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid ranking on semantic and KG candidates.

    Args:
        semantic_candidates: List of semantic results from VDB
        kg_candidates: List of KG results from Neo4j
        query_entities: Entities extracted from original post
        top_k: Number of top results to return
        round_id: Round identifier for logging

    Returns:
        List of ranked candidates sorted by final_score (descending)
    """
    ranked = hybrid_rank(semantic_candidates, kg_candidates, query_entities=query_entities)
    top_ranked = ranked[:top_k]

    score_str = top_ranked[0]["final_score"] if top_ranked else "N/A"
    logger.info(
        f"[RankingPhase:{round_id}] Ranked {len(ranked)} candidates, "
        f"top score: {score_str}, returned {len(top_ranked)} top-k"
    )

    return top_ranked
