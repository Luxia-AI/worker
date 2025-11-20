"""
Ingestion Phase: Persist facts and triples to VDB and KG.
"""

from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.kg.kg_ingest import KGIngest
from app.services.vdb.vdb_ingest import VDBIngest

logger = get_logger(__name__)


async def ingest_facts_and_triples(
    vdb_ingest: VDBIngest,
    kg_ingest: KGIngest,
    facts: List[Dict[str, Any]],
    triples: List[Dict[str, Any]],
    round_id: str,
) -> None:
    """
    Ingest facts to VDB and triples to KG (non-blocking best-effort).

    Args:
        vdb_ingest: VDBIngest instance (Pinecone)
        kg_ingest: KGIngest instance (Neo4j)
        facts: List of fact dicts to ingest
        triples: List of triple dicts to ingest
        round_id: Round identifier for logging
    """
    # Ingest facts to VDB
    if facts:
        try:
            await vdb_ingest.embed_and_ingest(facts)
            logger.info(f"[IngestionPhase:{round_id}] Ingested {len(facts)} facts to VDB")
        except Exception as e:
            logger.warning(f"[IngestionPhase:{round_id}] VDB ingest failed: {e}")

    # Ingest triples to KG
    if triples:
        try:
            await kg_ingest.ingest_triples(triples)
            logger.info(f"[IngestionPhase:{round_id}] Ingested {len(triples)} triples to KG")
        except Exception as e:
            logger.warning(f"[IngestionPhase:{round_id}] KG ingest failed: {e}")
