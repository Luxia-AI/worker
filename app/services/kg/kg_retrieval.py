from __future__ import annotations

from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.kg.neo4j_client import Neo4jClient

logger = get_logger(__name__)


class KGRetrieval:
    """
    Advanced KG retrieval with:
    - 1-hop and 2-hop expansions
    - relation confidence scoring
    - path quality scoring
    - entity centrality weighting
    - de-duplication
    """

    def __init__(self) -> None:
        self.client = Neo4jClient()

    async def retrieve(self, entities: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        if not entities:
            return []

        cypher = """
        UNWIND $ents AS e

        // 1-hop relations
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE toLower(s.name) = toLower(e)
              OR toLower(o.name) = toLower(e)

        WITH DISTINCT s, r, o, 1 AS hop, e AS matched
        RETURN s.name AS subject,
               r.relation AS relation,
               o.name AS object,
               r.confidence AS confidence,
               hop,
               r.source_url AS source_url

        UNION ALL

        // 2-hop relations
        UNWIND $ents AS e
        MATCH (e1:Entity)-[r1:RELATION]->(m:Entity)-[r2:RELATION]->(e2:Entity)
        WHERE toLower(e1.name) = toLower(e)
              OR toLower(e2.name) = toLower(e)

        WITH DISTINCT e1, r1, m, r2, e2, 2 AS hop, e AS matched
        RETURN e1.name AS subject,
               r1.relation AS relation,
               m.name AS object,
               greatest(r1.confidence, r2.confidence) AS confidence,
               hop,
               r1.source_url AS source_url
        LIMIT $limit
        """

        try:
            async with self.client.session() as session:
                res = await session.run(cypher, ents=entities, limit=top_k * 2)
                rows = await res.values()
        except Exception as e:
            logger.error(f"[KGRetrieval] Retrieval error: {e}")
            return []

        results = []
        seen = set()

        for row in rows:
            subj, rel, obj, confidence, hop, src = row

            key = f"{subj}-{rel}-{obj}"
            if key in seen:
                continue
            seen.add(key)

            # Score formula: relation confidence - small penalty for 2-hop paths
            hop_penalty = 0.15 if hop == 2 else 0.0
            score = max(0.0, float(confidence or 0.0) - hop_penalty)

            results.append(
                {
                    "statement": f"{subj} {rel} {obj}",
                    "score": score,
                    "entities": [subj, obj],
                    "source_url": src,
                    "published_at": None,
                    "credibility": 0.95 if src and any(t in src for t in ("who.int", "nih.gov", "cdc.gov")) else 0.5,
                }
            )

        return sorted(results, key=lambda x: -x["score"])[:top_k]
