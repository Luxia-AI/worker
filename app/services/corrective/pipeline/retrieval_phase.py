"""
Retrieval Phase: Retrieve semantic and KG candidates.
"""

import asyncio
import math
import re
from hashlib import sha1
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger, is_debug_enabled, log_value_payload
from app.services.common.dedup import dedup_candidates_by_score
from app.services.embedding.model import embed_async
from app.services.kg.kg_retrieval import KGRetrieval
from app.services.logging.log_manager import LogManager
from app.services.retrieval.lexical_index import LexicalIndex
from app.services.vdb.vdb_retrieval import VDBRetrieval, normalize_query_for_embedding

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


def _phrase_present(candidate_tokens: set[str], phrase: str) -> bool:
    phrase_tokens = _tokenize(phrase)
    if not phrase_tokens:
        return False
    overlap = len(candidate_tokens & phrase_tokens)
    if len(phrase_tokens) == 1:
        return overlap >= 1
    return overlap >= max(1, len(phrase_tokens) - 1)


def _strong_anchor_match(candidate_tokens: set[str], anchors: List[str]) -> bool:
    generic_singletons = {
        "health",
        "healthy",
        "immune",
        "immunity",
        "vitamin",
        "supplement",
        "supplements",
        "disease",
        "condition",
    }
    for anchor in anchors or []:
        a = str(anchor or "").strip().lower()
        if not a:
            continue
        toks = _tokenize(a)
        if not toks:
            continue
        if len(toks) == 1 and next(iter(toks), "") in generic_singletons:
            continue
        if _phrase_present(candidate_tokens, a):
            return True
    return False


def _normalize_retrieval_anchors(anchors: List[str]) -> List[str]:
    generic_singletons = {
        "health",
        "healthy",
        "immune",
        "immunity",
        "vitamin",
        "supplement",
        "supplements",
        "disease",
        "condition",
    }
    normalized: List[str] = []
    for anchor in anchors or []:
        clean = str(anchor or "").strip().lower()
        if clean and clean not in normalized:
            normalized.append(clean)
    token_sets = [(a, _tokenize(a)) for a in normalized]
    filtered: List[str] = []
    for anchor, toks in token_sets:
        if not toks:
            continue
        if len(toks) == 1:
            tok = next(iter(toks), "")
            has_more_specific = any(other != anchor and toks < other_toks for other, other_toks in token_sets)
            if tok in generic_singletons and (len(token_sets) > 1 or has_more_specific):
                continue
            if has_more_specific:
                continue
        filtered.append(anchor)
    return filtered or normalized


def _claim_context_hash(claim_text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(claim_text or "").strip().lower())
    if not normalized:
        return ""
    return sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _kg_fallback_entities(query_text: str, anchors: List[str], entities: List[str]) -> List[str]:
    """
    Generic KG fallback expansion:
    - keep normalized anchors/entities
    - add noun-phrase chunks from claim text (non domain-specific)
    """
    out: List[str] = []
    seen = set()
    for source in (anchors or []) + (entities or []):
        e = str(source or "").strip().lower()
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    # noun phrase fallback (up to trigrams)
    tokens = [t for t in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", (query_text or "").lower())]
    stop = {
        "the",
        "and",
        "or",
        "with",
        "without",
        "from",
        "into",
        "that",
        "this",
        "those",
        "these",
        "may",
        "might",
        "could",
        "should",
        "would",
        "health",
    }
    clean = [t for t in tokens if t not in stop]
    for n in (3, 2, 1):
        for i in range(0, max(0, len(clean) - n + 1)):
            phrase = " ".join(clean[i : i + n]).strip()
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            out.append(phrase)
            if len(out) >= 10:
                return out
    return out[:10]


def _semantic_candidate_anchor_overlap(candidate: Dict[str, Any], anchors: List[str], claim_tokens: set[str]) -> bool:
    statement = str(candidate.get("statement") or "")
    entities = " ".join(str(e) for e in (candidate.get("entities") or []))
    ctx_entities = " ".join(str(e) for e in (candidate.get("claim_context_entities") or []))
    cand_tokens = _tokenize(" ".join([statement, entities, ctx_entities]))
    if not cand_tokens:
        return False

    if _strong_anchor_match(cand_tokens, anchors):
        return True
    generic_claim_tokens = {
        "health",
        "healthy",
        "immune",
        "immunity",
        "vitamin",
        "supplement",
        "supplements",
        "support",
        "function",
        "benefit",
        "benefits",
        "effect",
        "effects",
    }
    claim_specific_tokens = {t for t in claim_tokens if t not in generic_claim_tokens}
    if claim_specific_tokens and (cand_tokens & claim_specific_tokens):
        return True
    if not claim_specific_tokens and claim_tokens and (cand_tokens & claim_tokens):
        return True
    return False


def _kg_candidate_anchor_overlap(candidate: Dict[str, Any], anchors: List[str], claim_tokens: set[str]) -> bool:
    subject = str(candidate.get("subject") or "")
    object_ = str(candidate.get("object") or "")
    statement = str(candidate.get("statement") or "")
    relation = str(candidate.get("relation") or "")
    cand_tokens = _tokenize(" ".join([subject, object_, relation, statement]))
    if not cand_tokens:
        return False

    if _strong_anchor_match(cand_tokens, anchors):
        return True
    generic_claim_tokens = {
        "health",
        "healthy",
        "immune",
        "immunity",
        "vitamin",
        "supplement",
        "supplements",
        "support",
        "function",
        "benefit",
        "benefits",
        "effect",
        "effects",
    }
    claim_specific_tokens = {t for t in claim_tokens if t not in generic_claim_tokens}
    if claim_specific_tokens and (cand_tokens & claim_specific_tokens):
        return True
    if not claim_specific_tokens and claim_tokens and (cand_tokens & claim_tokens):
        return True
    return False


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
    claim_anchors: Optional[List[str]] = None,
    include_metrics: bool = False,
    claim_context_hash: str = "",
) -> (
    tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    | tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]
):
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
    normalized_queries: List[str] = []
    anchors = [str(a).strip() for a in (claim_anchors or all_entities or []) if str(a).strip()]
    anchors = _normalize_retrieval_anchors(anchors)
    claim_tokens = _tokenize(query_text)
    effective_claim_context_hash = str(claim_context_hash or "").strip().lower() or _claim_context_hash(query_text)

    if not topics:
        logger.warning(
            f"[RetrievalPhase:{round_id}] No topics provided; using no-topic fallback for VDB/BM25 retrieval."
        )

    # Run KG retrieval concurrently with semantic retrieval loop.
    kg_task = asyncio.create_task(
        kg_retriever.retrieve(
            all_entities,
            top_k=top_k,
            query_text=query_text,
            claim_anchors=anchors,
            claim_context_hash=effective_claim_context_hash,
        )
    )

    for q in queries:
        q_embed = normalize_query_for_embedding(q)
        if not q_embed:
            continue
        normalized_queries.append(q_embed)
        try:
            sem_res = await vdb_retriever.search(
                q_embed,
                top_k=top_k,
                topics=topics or None,
                claim_text=query_text,
                claim_anchors=anchors,
                claim_context_hash=effective_claim_context_hash,
            )
            semantic_candidates.extend(sem_res or [])
        except Exception as e:
            logger.warning(f"[RetrievalPhase:{round_id}] VDB retrieval failed for query='{q_embed}': {e}")

            if log_manager:
                await log_manager.add_log(
                    level="WARNING",
                    message=f"VDB retrieval failed for query: {q_embed}",
                    module=__name__,
                    request_id=f"claim-{round_id}",
                    round_id=round_id,
                    context={"query": q_embed, "error": str(e)},
                )

        if lexical_index:
            try:
                bm25_hits = lexical_index.search(q_embed, topics=topics or None)
                for hit in bm25_hits:
                    fact_id = hit.get("fact_id")
                    bm25 = float(hit.get("bm25") or 0.0)
                    if fact_id:
                        existing = bm25_ids.get(fact_id)
                        if existing is None or bm25 < existing:
                            bm25_ids[fact_id] = bm25
            except Exception as e:
                logger.warning(f"[RetrievalPhase:{round_id}] BM25 retrieval failed for query='{q_embed}': {e}")

                if log_manager:
                    await log_manager.add_log(
                        level="WARNING",
                        message=f"BM25 retrieval failed for query: {q_embed}",
                        module=__name__,
                        request_id=f"claim-{round_id}",
                        round_id=round_id,
                        context={"query": q_embed, "error": str(e)},
                    )

    # Re-rank BM25 shortlist with Pinecone vectors (if available)
    if bm25_ids:
        try:
            if normalized_queries and not query_embeddings:
                try:
                    query_embeddings = await embed_async(normalized_queries)
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

    if anchors and semantic_candidates:
        filtered_sem: List[Dict[str, Any]] = []
        sem_dropped_by_anchor = 0
        for c in semantic_candidates:
            claim_ctx_match = bool(c.get("claim_context_match"))
            overlap_ok = _semantic_candidate_anchor_overlap(c, anchors, claim_tokens)
            if claim_ctx_match or overlap_ok:
                filtered_sem.append(c)
                continue
            score = float(c.get("score") or 0.0)
            # Keep very strong semantic matches as conservative fallback.
            if score >= 0.72:
                filtered_sem.append(c)
                continue
            sem_dropped_by_anchor += 1
        if sem_dropped_by_anchor > 0:
            logger.info(
                "[RetrievalPhase:%s] Dropped %d semantic candidates failing claim-anchor overlap prefilter",
                round_id,
                sem_dropped_by_anchor,
            )
        semantic_candidates = filtered_sem

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
    log_value_payload(
        logger,
        "retrieval",
        {
            "round_id": round_id,
            "semantic_raw_count": len(semantic_candidates or []),
            "semantic_dedup_count": len(dedup_sem or []),
            "semantic_candidates_sample": [
                {
                    "statement": c.get("statement", ""),
                    "source_url": c.get("source_url", ""),
                    "score": c.get("score", 0.0),
                    "candidate_type": c.get("candidate_type", "VDB"),
                }
                for c in dedup_sem
            ],
        },
    )

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"[PhaseOutput] retrieval semantic_dedup_count={len(dedup_sem)}",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={"dedup_candidates": len(dedup_sem), "raw_candidates": len(semantic_candidates)},
        )

    # KG retrieval
    kg_candidates = []
    kg_fallback_triggered = False
    try:
        kg_candidates = await kg_task
    except Exception as e:
        logger.warning(f"[RetrievalPhase:{round_id}] KG retrieval failed: {e}")

    if not kg_candidates:
        fallback_entities = _kg_fallback_entities(query_text=query_text, anchors=anchors, entities=all_entities)
        if fallback_entities and fallback_entities != (all_entities or []):
            kg_fallback_triggered = True
            logger.info(
                "[RetrievalPhase:%s] KG fallback triggered with %d expanded entities",
                round_id,
                len(fallback_entities),
            )
            try:
                kg_candidates = await kg_retriever.retrieve(
                    fallback_entities,
                    top_k=top_k,
                    query_text=query_text,
                    claim_anchors=anchors,
                    claim_context_hash=effective_claim_context_hash,
                )
            except Exception as e:
                logger.warning(f"[RetrievalPhase:{round_id}] KG fallback retrieval failed: {e}")

    if anchors and kg_candidates:
        filtered_kg: List[Dict[str, Any]] = []
        for c in kg_candidates:
            if _kg_candidate_anchor_overlap(c, anchors, claim_tokens):
                filtered_kg.append(c)
        dropped = len(kg_candidates) - len(filtered_kg)
        if dropped > 0:
            logger.info(
                "[RetrievalPhase:%s] Dropped %d KG candidates failing claim-anchor overlap prefilter",
                round_id,
                dropped,
            )
        kg_candidates = filtered_kg

    if not kg_candidates and anchors:
        logger.info(
            "[RetrievalPhase:%s] Skipping KG branch for this round: no KG candidates overlap claim anchors",
            round_id,
        )

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

    log_value_payload(
        logger,
        "retrieval",
        {
            "round_id": round_id,
            "kg_candidates_count": len(kg_candidates or []),
            "kg_with_score": kg_with_positive_score,
            "kg_max_score": round(kg_max_score, 4),
            "kg_candidates_sample": [
                {
                    "statement": c.get("statement", ""),
                    "source_url": c.get("source_url", ""),
                    "kg_score": c.get("kg_score", 0.0),
                }
                for c in kg_candidates
            ],
        },
    )
    kg_with_text = sum(1 for c in kg_candidates if (c.get("statement") or "").strip())
    kg_with_source = sum(1 for c in kg_candidates if (c.get("source_url") or "").strip())
    log_value_payload(
        logger,
        "retrieval",
        {
            "round_id": round_id,
            "kg_textualized": kg_with_text,
            "kg_with_source": kg_with_source,
            "kg_with_positive_score": kg_with_positive_score,
            "kg_max_score": round(kg_max_score, 4),
            "kg_fallback_triggered": bool(kg_fallback_triggered),
            "kg_zero_signal": bool(len(kg_candidates) == 0),
        },
        level="debug" if is_debug_enabled() else "info",
    )

    metrics = {
        "sem_raw": len(semantic_candidates),
        "sem_filtered": len(dedup_sem),
        "kg_raw": len(kg_candidates),
        "kg_with_score": kg_with_positive_score,
        "kg_zero_signal": 1 if len(kg_candidates) == 0 else 0,
        "kg_fallback_triggered": 1 if kg_fallback_triggered else 0,
    }
    if include_metrics:
        return dedup_sem, kg_candidates, metrics
    return dedup_sem, kg_candidates
