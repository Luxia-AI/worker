"""
Retrieval Phase: Retrieve semantic and KG candidates.
"""

from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.common.dedup import dedup_candidates_by_score
from app.services.kg.kg_retrieval import KGRetrieval
from app.services.vdb.vdb_retrieval import VDBRetrieval

logger = get_logger(__name__)


async def retrieve_candidates(
    vdb_retriever: VDBRetrieval,
    kg_retriever: KGRetrieval,
    queries: List[str],
    all_entities: List[str],
    top_k: int,
    round_id: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve semantic and KG candidates.

    Args:
        vdb_retriever: VDBRetrieval instance (Pinecone)
        kg_retriever: KGRetrieval instance (Neo4j)
        queries: List of search queries
        all_entities: List of extracted entities
        top_k: Number of top candidates to return
        round_id: Round identifier for logging

    Returns:
        Tuple of (dedup_semantic_candidates, kg_candidates)
    """
    # Semantic retrieval using VDB
    semantic_candidates = []
    for q in queries:
        try:
            sem_res = await vdb_retriever.search(q, top_k=top_k)
            semantic_candidates.extend(sem_res or [])
        except Exception as e:
            logger.warning(f"[RetrievalPhase:{round_id}] VDB retrieval failed for query='{q}': {e}")

    # Deduplicate semantic candidates
    dedup_sem = dedup_candidates_by_score(
        semantic_candidates,
        statement_key="statement",
        source_key="source_url",
        score_key="score",
    )
    logger.info(
        f"[RetrievalPhase:{round_id}] Retrieved {len(dedup_sem)} semantic candidates "
        f"(from {len(semantic_candidates)} raw)"
    )

    # KG retrieval
    kg_candidates = []
    try:
        kg_candidates = await kg_retriever.retrieve(all_entities, top_k=top_k)
    except Exception as e:
        logger.warning(f"[RetrievalPhase:{round_id}] KG retrieval failed: {e}")

    logger.info(f"[RetrievalPhase:{round_id}] Retrieved {len(kg_candidates)} KG candidates")

    return dedup_sem, kg_candidates
