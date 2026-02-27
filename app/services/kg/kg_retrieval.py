from __future__ import annotations

import asyncio
import re
from hashlib import sha1
from typing import Any, Dict, List, Optional

from app.config.trusted_domains import is_trusted_domain
from app.core.logger import get_logger
from app.services.kg.neo4j_client import Neo4jClient

logger = get_logger(__name__)

# Timeout for KG retrieval operations
KG_RETRIEVAL_TIMEOUT = 15  # seconds


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


def _claim_context_hash(claim_text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(claim_text or "").strip().lower())
    if not normalized:
        return ""
    return sha1(normalized.encode("utf-8")).hexdigest()[:16]


class KGRetrieval:
    """
    Advanced KG retrieval with:
    - 1-hop and 2-hop relation expansions
    - Path quality scoring (relation confidence + hop penalty)
    - Credibility inference from source domain authority
    - Structured triple format (subject, relation, object)
    - Graceful timeout handling when Neo4j is unavailable
    """

    def __init__(self) -> None:
        self.client = Neo4jClient()

    async def retrieve(
        self,
        entities: List[str],
        top_k: int = 20,
        query_text: str = "",
        claim_anchors: Optional[List[str]] = None,
        claim_context_hash: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge graph paths connecting query entities.

        Args:
            entities: List of entity names to search for in KG
            top_k: Maximum number of results to return

        Returns:
            List of relation dicts with keys:
                - statement: str (formatted as "subject relation object")
                - score: float [0, 1] (path quality score)
                - entities: List[str] ([subject, object])
                - source_url: Optional[str]
                - published_at: Optional[str] (None for KG relations)
                - credibility: float [0, 1] (inferred from source domain)
                - hop_distance: int (1 or 2)
                - relation: str (relation type)
                - subject: str
                - object: str
                - confidence: float (relation confidence from KG)
        """
        if not entities:
            return []

        # Wrap in timeout to prevent blocking when Neo4j is unavailable
        try:
            return await asyncio.wait_for(
                self._do_retrieve(
                    entities,
                    top_k,
                    query_text=query_text,
                    claim_anchors=claim_anchors,
                    claim_context_hash=claim_context_hash,
                ),
                timeout=KG_RETRIEVAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[KGRetrieval] Timeout ({KG_RETRIEVAL_TIMEOUT}s), returning empty results")
            return []
        except ConnectionError as e:
            logger.warning(f"[KGRetrieval] Neo4j unavailable: {e}")
            return []
        except Exception as e:
            logger.error(f"[KGRetrieval] Unexpected error: {e}")
            return []

    async def _do_retrieve(
        self,
        entities: List[str],
        top_k: int,
        query_text: str = "",
        claim_anchors: Optional[List[str]] = None,
        claim_context_hash: str = "",
    ) -> List[Dict[str, Any]]:
        """Internal method that performs the actual retrieval."""
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
        normalized_entities: List[str] = []
        for e in entities or []:
            clean = str(e or "").strip().lower()
            if clean and clean not in normalized_entities:
                normalized_entities.append(clean)
        token_sets = [(e, _tokenize(e)) for e in normalized_entities]
        retrieval_entities: List[str] = []
        for ent, toks in token_sets:
            if not toks:
                continue
            if len(toks) == 1:
                tok = next(iter(toks), "")
                has_more_specific = any(other_ent != ent and toks < other_toks for other_ent, other_toks in token_sets)
                if tok in generic_singletons and (len(token_sets) > 1 or has_more_specific):
                    continue
                if has_more_specific:
                    continue
            retrieval_entities.append(ent)
        if not retrieval_entities:
            retrieval_entities = normalized_entities[:]

        cypher = """
        UNWIND $ents AS e

        // 1-hop relations: (Entity)-[:SUBJECT_OF]->(Relation)-[:OBJECT_OF]->(Entity)
        MATCH (s:Entity)-[:SUBJECT_OF]->(rel:Relation)-[:OBJECT_OF]->(o:Entity)
        WHERE toLower(s.name) = toLower(e)
              OR toLower(o.name) = toLower(e)
              OR (size(e) > 3 AND toLower(s.name) CONTAINS toLower(e))
              OR (size(e) > 3 AND toLower(o.name) CONTAINS toLower(e))
        OPTIONAL MATCH (rel)-[:SUPPORTED_BY]->(src:Source)

        WITH DISTINCT s, rel, o, 1 AS hop, src.url AS src_url
        RETURN s.name AS subject,
               rel.predicate AS relation,
               o.name AS object,
               rel.confidence AS confidence,
               hop,
               src_url AS source_url,
               coalesce(rel.claim_context_hash, "") AS claim_context_hash,
               coalesce(rel.claim_context_entities, []) AS claim_context_entities

        UNION ALL

        // 2-hop relations via intermediate entity
        UNWIND $ents AS e
        MATCH (e1:Entity)-[:SUBJECT_OF]->(rel1:Relation)-[:OBJECT_OF]->(m:Entity)
              -[:SUBJECT_OF]->(rel2:Relation)-[:OBJECT_OF]->(e2:Entity)
        WHERE toLower(e1.name) = toLower(e)
              OR toLower(e2.name) = toLower(e)
              OR (size(e) > 3 AND toLower(e1.name) CONTAINS toLower(e))
              OR (size(e) > 3 AND toLower(e2.name) CONTAINS toLower(e))
        OPTIONAL MATCH (rel1)-[:SUPPORTED_BY]->(src1:Source)
        OPTIONAL MATCH (rel2)-[:SUPPORTED_BY]->(src2:Source)

        WITH DISTINCT e1, rel1, m, rel2, e2, 2 AS hop,
             COALESCE(src1.url, src2.url) AS src_url,
             CASE
               WHEN $claim_context_hash <> '' AND coalesce(rel1.claim_context_hash, '') = $claim_context_hash
                    THEN coalesce(rel1.claim_context_hash, '')
               WHEN $claim_context_hash <> '' AND coalesce(rel2.claim_context_hash, '') = $claim_context_hash
                    THEN coalesce(rel2.claim_context_hash, '')
               WHEN rel1.confidence >= rel2.confidence THEN coalesce(rel1.claim_context_hash, '')
               ELSE coalesce(rel2.claim_context_hash, '')
             END AS claim_ctx_hash,
             CASE
               WHEN $claim_context_hash <> '' AND coalesce(rel1.claim_context_hash, '') = $claim_context_hash
                    THEN coalesce(rel1.claim_context_entities, [])
               WHEN $claim_context_hash <> '' AND coalesce(rel2.claim_context_hash, '') = $claim_context_hash
                    THEN coalesce(rel2.claim_context_entities, [])
               WHEN rel1.confidence >= rel2.confidence THEN coalesce(rel1.claim_context_entities, [])
               ELSE coalesce(rel2.claim_context_entities, [])
             END AS claim_ctx_entities
        RETURN e1.name AS subject,
               rel1.predicate AS relation,
               m.name AS object,
               CASE WHEN rel1.confidence > rel2.confidence THEN rel1.confidence ELSE rel2.confidence END AS confidence,
               hop,
               src_url AS source_url,
               claim_ctx_hash AS claim_context_hash,
               claim_ctx_entities AS claim_context_entities
        LIMIT $limit
        """

        try:
            async with self.client.session() as session:
                res = await session.run(
                    cypher,
                    ents=retrieval_entities,
                    limit=top_k * 2,
                    claim_context_hash=str(claim_context_hash or "").strip().lower(),
                )
                rows = await res.values()
        except Exception as e:
            logger.error(f"[KGRetrieval] Query execution failed: {e}")
            return []

        results = []
        seen = set()
        query_keywords = _tokenize(query_text)
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
        query_specific_keywords = {t for t in query_keywords if t not in generic_claim_tokens}
        anchors = [str(a).strip().lower() for a in (claim_anchors or retrieval_entities or []) if str(a).strip()]
        anchor_tokens: set[str] = set()
        for anchor in anchors:
            anchor_tokens |= _tokenize(anchor)
        normalized_claim_hash = str(claim_context_hash or "").strip().lower() or _claim_context_hash(query_text)

        for row in rows:
            subj, rel, obj, confidence, hop, src, row_claim_hash, row_claim_entities = row

            key = f"{subj}-{rel}-{obj}"
            if key in seen:
                continue
            seen.add(key)

            # Path quality score: relation confidence - penalty for longer paths
            # 1-hop paths are trusted more than 2-hop (less transitive loss)
            hop_penalty = 0.15 if hop == 2 else 0.0
            path_score = max(0.0, float(confidence or 0.0) - hop_penalty)

            # Infer credibility from source domain authority
            credibility = self._infer_credibility(src)

            rel_norm = (rel or "").replace("_", " ").strip()
            statement = f"{subj} {rel_norm} {obj}".strip()
            claim_entities = row_claim_entities if isinstance(row_claim_entities, list) else []
            claim_entities = [str(e).strip().lower() for e in claim_entities if str(e).strip()][:20]
            statement_tokens = _tokenize(
                " ".join(
                    [
                        statement,
                        " ".join(claim_entities),
                    ]
                )
            )
            query_overlap = len(statement_tokens & query_keywords) if query_keywords else 0
            query_specific_overlap = len(statement_tokens & query_specific_keywords) if query_specific_keywords else 0
            anchor_overlap = len(statement_tokens & anchor_tokens) if anchor_tokens else 0
            strong_anchor = _strong_anchor_match(statement_tokens, anchors)
            row_claim_hash_norm = str(row_claim_hash or "").strip().lower()
            claim_context_match = bool(
                normalized_claim_hash and row_claim_hash_norm and row_claim_hash_norm == normalized_claim_hash
            )

            if (
                anchor_tokens
                and not claim_context_match
                and not strong_anchor
                and anchor_overlap == 0
                and query_specific_overlap == 0
                and (not query_specific_keywords and query_overlap == 0)
            ):
                # Reject obvious drift in low-signal relations.
                if hop > 1 or path_score < 0.65:
                    continue

            if claim_context_match:
                path_score = min(1.0, path_score + 0.20)
            if anchor_overlap > 0:
                path_score = min(1.0, path_score + min(0.15, 0.03 * anchor_overlap))
            if query_specific_overlap > 0:
                path_score = min(1.0, path_score + min(0.08, 0.02 * query_specific_overlap))
            elif not query_specific_keywords and query_overlap > 0:
                path_score = min(1.0, path_score + min(0.05, 0.01 * query_overlap))

            result = {
                "statement": statement,
                "score": path_score,  # hybrid ranker score
                "kg_score_raw": path_score,
                "kg_score": path_score,
                "entities": [subj, obj, rel_norm],
                "source_url": src,
                "published_at": None,  # KG relations don't have publish dates
                "credibility": credibility,
                "candidate_type": "KG",
                # Additional structured fields for transparency
                "subject": subj,
                "relation": rel,
                "object": obj,
                "confidence": float(confidence or 0.0),  # raw relation confidence
                "hop_distance": hop,
                "path_quality_score": path_score,
                "claim_context_hash": row_claim_hash_norm,
                "claim_context_match": claim_context_match,
                "claim_context_entities": claim_entities,
                "anchor_overlap_count": anchor_overlap,
                "claim_overlap_count": query_overlap,
            }
            results.append(result)

        # DETERMINISTIC ORDERING: Sort by score DESC, then statement ASC for consistent results
        # This ensures identical queries return identical ordering even when scores are equal
        sorted_results = sorted(results, key=lambda x: (-x["score"], x["statement"]))[:top_k]
        logger.info(
            f"[KGRetrieval] Retrieved {len(sorted_results)} relations from {len(retrieval_entities)} entities "
            f"(raw matches: {len(results)})"
        )
        return sorted_results

    @staticmethod
    def _infer_credibility(source_url: str | None) -> float:
        """
        Infer credibility score from source domain.

        Args:
            source_url: Source URL from KG relation

        Returns:
            Credibility score [0, 1]
        """
        if not source_url:
            return 0.5  # Unknown source gets neutral credibility

        source_url_lower = source_url.lower()

        # Canonical trusted domains.
        if is_trusted_domain(source_url):
            return 0.95

        # General news/media (lower credibility)
        if any(domain in source_url_lower for domain in ("news", "press", "blog", "medium.com")):
            return 0.40

        # Unknown source
        return 0.5
