from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.kg.neo4j_client import Neo4jClient

logger = get_logger(__name__)

# Timeout for KG retrieval operations
KG_RETRIEVAL_TIMEOUT = 15  # seconds


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

    async def retrieve(self, entities: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
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
            return await asyncio.wait_for(self._do_retrieve(entities, top_k), timeout=KG_RETRIEVAL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"[KGRetrieval] Timeout ({KG_RETRIEVAL_TIMEOUT}s), returning empty results")
            return []
        except ConnectionError as e:
            logger.warning(f"[KGRetrieval] Neo4j unavailable: {e}")
            return []
        except Exception as e:
            logger.error(f"[KGRetrieval] Unexpected error: {e}")
            return []

    async def _do_retrieve(self, entities: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Internal method that performs the actual retrieval."""
        cypher = """
        UNWIND $ents AS e

        // 1-hop relations
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE toLower(s.name) = toLower(e)
              OR toLower(o.name) = toLower(e)

        WITH DISTINCT s, r, o, 1 AS hop, e AS matched, r.source_url AS src_url
        RETURN s.name AS subject,
               r.relation AS relation,
               o.name AS object,
               r.confidence AS confidence,
               hop,
               src_url AS source_url

        UNION ALL

        // 2-hop relations (with path reconstruction)
        UNWIND $ents AS e
        MATCH (e1:Entity)-[r1:RELATION]->(m:Entity)-[r2:RELATION]->(e2:Entity)
        WHERE toLower(e1.name) = toLower(e)
              OR toLower(e2.name) = toLower(e)

        WITH DISTINCT e1, r1, m, r2, e2, 2 AS hop, e AS matched
        RETURN e1.name AS subject,
               r1.relation AS relation,
               m.name AS object,
               CASE WHEN r1.confidence > r2.confidence THEN r1.confidence ELSE r2.confidence END AS confidence,
               hop,
               COALESCE(r1.source_url, r2.source_url) AS source_url
        LIMIT $limit
        """

        try:
            async with self.client.session() as session:
                res = await session.run(cypher, ents=entities, limit=top_k * 2)
                rows = await res.values()
        except Exception as e:
            logger.error(f"[KGRetrieval] Query execution failed: {e}")
            return []

        results = []
        seen = set()

        for row in rows:
            subj, rel, obj, confidence, hop, src = row

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

            result = {
                "statement": f"{subj} {rel} {obj}",
                "score": path_score,  # hybrid ranker score
                "entities": [subj, obj],
                "source_url": src,
                "published_at": None,  # KG relations don't have publish dates
                "credibility": credibility,
                # Additional structured fields for transparency
                "subject": subj,
                "relation": rel,
                "object": obj,
                "confidence": float(confidence or 0.0),  # raw relation confidence
                "hop_distance": hop,
                "path_quality_score": path_score,
            }
            results.append(result)

        sorted_results = sorted(results, key=lambda x: -x["score"])[:top_k]
        logger.info(
            f"[KGRetrieval] Retrieved {len(sorted_results)} relations from {len(entities)} entities "
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

        # Very high-authority medical domains
        if any(domain in source_url_lower for domain in ("who.int", "nih.gov", "cdc.gov")):
            return 0.95

        # Government/educational domains
        if any(domain in source_url_lower for domain in (".gov", ".edu")):
            return 0.75

        # Trusted medical news/publishers
        if any(domain in source_url_lower for domain in ("health.harvard.edu", "medlineplus.gov", "nhs.uk")):
            return 0.85

        # General news/media (lower credibility)
        if any(domain in source_url_lower for domain in ("news", "press", "blog", "medium.com")):
            return 0.40

        # Unknown source
        return 0.5
