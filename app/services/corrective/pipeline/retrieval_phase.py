"""
Retrieval Phase: Retrieve semantic and KG candidates.
"""

import math
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.services.common.dedup import dedup_candidates_by_score
from app.services.embedding.model import embed_async
from app.services.kg.kg_retrieval import KGRetrieval
from app.services.logging.log_manager import LogManager
from app.services.retrieval.lexical_index import LexicalIndex
from app.services.vdb.vdb_retrieval import VDBRetrieval

logger = get_logger(__name__)


async def retrieve_candidates(
    vdb_retriever: VDBRetrieval,
    kg_retriever: KGRetrieval,
    queries: List[str],
    all_entities: List[str],
    top_k: int,
    round_id: str,
    topics: List[str],
    lexical_index: Optional[LexicalIndex] = None,
    log_manager: Optional[LogManager] = None,
    query_text: str = "",
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
        topics: Required topic filter for VDB/BM25 retrieval
        lexical_index: Optional BM25 lexical index

    Returns:
        Tuple of (dedup_semantic_candidates, kg_candidates)
    """
    # Semantic retrieval using VDB
    semantic_candidates = []
    bm25_ids: Dict[str, float] = {}
    query_embeddings: List[List[float]] = []

    if not topics:
        logger.warning(f"[RetrievalPhase:{round_id}] No topics provided; skipping VDB retrieval.")

    for q in queries:
        if topics:
            try:
                sem_res = await vdb_retriever.search(q, top_k=top_k, topics=topics)
                semantic_candidates.extend(sem_res or [])
            except Exception as e:
                logger.warning(f"[RetrievalPhase:{round_id}] VDB retrieval failed for query='{q}': {e}")

                if log_manager:
                    await log_manager.add_log(
                        level="WARNING",
                        message=f"VDB retrieval failed for query: {q}",
                        module=__name__,
                        request_id=f"claim-{round_id}",
                        round_id=round_id,
                        context={"query": q, "error": str(e)},
                    )

        if lexical_index and topics:
            try:
                bm25_hits = lexical_index.search(q, topics=topics)
                for hit in bm25_hits:
                    fact_id = hit.get("fact_id")
                    bm25 = float(hit.get("bm25") or 0.0)
                    if fact_id:
                        existing = bm25_ids.get(fact_id)
                        if existing is None or bm25 < existing:
                            bm25_ids[fact_id] = bm25
            except Exception as e:
                logger.warning(f"[RetrievalPhase:{round_id}] BM25 retrieval failed for query='{q}': {e}")

                if log_manager:
                    await log_manager.add_log(
                        level="WARNING",
                        message=f"BM25 retrieval failed for query: {q}",
                        module=__name__,
                        request_id=f"claim-{round_id}",
                        round_id=round_id,
                        context={"query": q, "error": str(e)},
                    )

    # Re-rank BM25 shortlist with Pinecone vectors (if available)
    if bm25_ids and topics:
        try:
            if queries and not query_embeddings:
                try:
                    query_embeddings = await embed_async(queries)
                except Exception as e:
                    logger.warning(f"[RetrievalPhase:{round_id}] Query embedding failed: {e}")
                    query_embeddings = []

            bm25_results = vdb_retriever.fetch_by_ids(list(bm25_ids.keys()), include_values=True)
            # Compute cosine similarity with each query embedding
            for item in bm25_results:
                vec = item.get("values")
                if not vec or not query_embeddings:
                    continue

                def _cos(a: List[float], b: List[float]) -> float:
                    dot = sum(x * y for x, y in zip(a, b))
                    na = math.sqrt(sum(x * x for x in a))
                    nb = math.sqrt(sum(y * y for y in b))
                    if na == 0.0 or nb == 0.0:
                        return 0.0
                    return dot / (na * nb)

                best = 0.0
                for qv in query_embeddings:
                    best = max(best, _cos(vec, qv))

                if best > 0.0:
                    item["score"] = best
                    item["bm25_score"] = bm25_ids.get(item.get("id", ""), 0.0)
                    semantic_candidates.append(item)
        except Exception as e:
            logger.warning(f"[RetrievalPhase:{round_id}] BM25 re-rank failed: {e}")

    # Deduplicate semantic candidates
    dedup_sem = dedup_candidates_by_score(
        semantic_candidates,
        statement_key="statement",
        source_key="source_url",
        score_key="score",
    )
    for c in dedup_sem:
        c["candidate_type"] = c.get("candidate_type") or "VDB"
        c["is_backfill"] = bool(c.get("is_backfill", False))
    logger.info(
        f"[RetrievalPhase:{round_id}] Retrieved {len(dedup_sem)} semantic candidates "
        f"(from {len(semantic_candidates)} raw)"
    )

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"Semantic retrieval completed: {len(dedup_sem)} candidates",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={"dedup_candidates": len(dedup_sem), "raw_candidates": len(semantic_candidates)},
        )

    # KG retrieval
    kg_candidates = []
    try:
        kg_candidates = await kg_retriever.retrieve(all_entities, top_k=top_k, query_text=query_text)
    except Exception as e:
        logger.warning(f"[RetrievalPhase:{round_id}] KG retrieval failed: {e}")

    kg_with_positive_score = 0
    kg_max_score = 0.0
    for c in kg_candidates:
        if not (c.get("statement") or "").strip():
            subj = (c.get("subject") or "").strip()
            rel = (c.get("relation") or "").replace("_", " ").strip()
            obj = (c.get("object") or "").strip()
            stmt = " ".join(p for p in [subj, rel, obj] if p).strip()
            c["statement"] = stmt
        raw = max(
            float(c.get("kg_score_raw") or 0.0),
            float(c.get("kg_score") or 0.0),
            float(c.get("confidence") or 0.0),
            float(c.get("path_quality_score") or 0.0),
            float(c.get("score") or 0.0),
        )
        # Small provenance lift when source exists.
        if (c.get("source_url") or "").strip():
            raw = min(1.0, raw + 0.05)
        c["kg_score_raw"] = raw
        c["kg_score"] = raw
        c["score"] = max(float(c.get("score") or 0.0), raw)
        c["candidate_type"] = "KG"
        if raw > 0.0:
            kg_with_positive_score += 1
        kg_max_score = max(kg_max_score, raw)

    logger.info(f"[RetrievalPhase:{round_id}] Retrieved {len(kg_candidates)} KG candidates")
    kg_with_text = sum(1 for c in kg_candidates if (c.get("statement") or "").strip())
    kg_with_source = sum(1 for c in kg_candidates if (c.get("source_url") or "").strip())
    logger.info(
        f"[RetrievalPhase:{round_id}] KG->evidence conversion: total={len(kg_candidates)}, "
        f"textualized={kg_with_text}, with_source={kg_with_source}, "
        f"kg_with_score={kg_with_positive_score}, max_kg_score={kg_max_score:.3f}"
    )

    return dedup_sem, kg_candidates
