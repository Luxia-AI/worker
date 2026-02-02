from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.kg.neo4j_client import NEO4J_QUERY_TIMEOUT, Neo4jClient

logger = get_logger(__name__)

# Overall timeout for the entire ingest operation
KG_INGEST_TIMEOUT = 30  # seconds


class KGIngest:
    """
    Handles ingestion of entity-relation-entity triples into Neo4j.

    Creates:
        (Entity {name})-[RELATION {confidence, source_url, published_at}]->(Entity)
    Ensures:
        - nodes are merged (idempotent)
        - relations upserted with confidence + metadata
        - logs every action
        - fails gracefully with timeout if Neo4j is unavailable
    """

    def __init__(self) -> None:
        self.client = Neo4jClient()

    async def ingest_triples(self, triples: List[Dict[str, Any]]) -> int:
        if not triples:
            return 0

        # Wrap entire operation in timeout to prevent blocking
        try:
            return await asyncio.wait_for(self._do_ingest(triples), timeout=KG_INGEST_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"[KGIngest] Overall timeout ({KG_INGEST_TIMEOUT}s), skipping KG ingestion")
            return 0
        except ConnectionError as e:
            logger.warning(f"[KGIngest] Neo4j unavailable, skipping: {e}")
            return 0
        except Exception as e:
            logger.error(f"[KGIngest] Unexpected error, skipping: {e}")
            return 0

    async def _do_ingest(self, triples: List[Dict[str, Any]]) -> int:
        """Internal method that performs the actual ingestion."""
        cypher = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATION {relation: $relation}]->(o)
        SET r.confidence = $confidence,
            r.source_url = $source_url,
            r.published_at = $published_at,
            r.updated_at = datetime()
        RETURN id(r) AS rel_id
        """

        count = 0

        try:
            async with self.client.session() as session:
                for triple in triples:
                    subj = triple.get("subject", "").strip()
                    rel = triple.get("relation", "").strip()
                    obj = triple.get("object", "").strip()

                    if not subj or not rel or not obj:
                        logger.warning(f"[KGIngest] Skipped malformed triple: {triple}")
                        continue

                    params = {
                        "subject": subj,
                        "object": obj,
                        "relation": rel,
                        "confidence": float(triple.get("confidence", 0.0)),
                        "source_url": triple.get("source_url"),
                        "published_at": triple.get("published_at"),
                    }

                    try:
                        # Add per-query timeout
                        await asyncio.wait_for(session.run(cypher, **params), timeout=NEO4J_QUERY_TIMEOUT)
                        count += 1
                    except asyncio.TimeoutError:
                        logger.warning(f"[KGIngest] Query timeout for triple: {subj}->{rel}->{obj}")
                        # Continue with other triples
                    except Exception as e:
                        logger.error(f"[KGIngest] Failed ingest triple {triple}: {e}")

        except ConnectionError as e:
            logger.warning(f"[KGIngest] Connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[KGIngest] Session failed: {e}")

        logger.info(f"[KGIngest] Successfully ingested {count} triples")
        return count
