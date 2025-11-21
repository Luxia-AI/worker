"""
Ingestion Phase: Persist facts and triples to VDB and KG.
"""

from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.services.kg.kg_ingest import KGIngest
from app.services.logging.log_manager import LogManager
from app.services.vdb.vdb_ingest import VDBIngest

logger = get_logger(__name__)


async def ingest_facts_and_triples(
    vdb_ingest: VDBIngest,
    kg_ingest: KGIngest,
    facts: List[Dict[str, Any]],
    triples: List[Dict[str, Any]],
    round_id: str,
    log_manager: Optional[LogManager] = None,
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

            if log_manager:
                await log_manager.add_log(
                    level="INFO",
                    message=f"VDB ingestion completed: {len(facts)} facts",
                    module=__name__,
                    request_id=f"claim-{round_id}",
                    round_id=round_id,
                    context={"facts_ingested": len(facts)},
                )
        except Exception as e:
            logger.warning(f"[IngestionPhase:{round_id}] VDB ingest failed: {e}")

            if log_manager:
                await log_manager.add_log(
                    level="WARNING",
                    message=f"VDB ingestion failed: {str(e)}",
                    module=__name__,
                    request_id=f"claim-{round_id}",
                    round_id=round_id,
                    context={"error": str(e)},
                )

    # Ingest triples to KG
    if triples:
        try:
            await kg_ingest.ingest_triples(triples)
            logger.info(f"[IngestionPhase:{round_id}] Ingested {len(triples)} triples to KG")

            if log_manager:
                await log_manager.add_log(
                    level="INFO",
                    message=f"KG ingestion completed: {len(triples)} triples",
                    module=__name__,
                    request_id=f"claim-{round_id}",
                    round_id=round_id,
                    context={"triples_ingested": len(triples)},
                )
        except Exception as e:
            logger.warning(f"[IngestionPhase:{round_id}] KG ingest failed: {e}")

            if log_manager:
                await log_manager.add_log(
                    level="WARNING",
                    message=f"KG ingestion failed: {str(e)}",
                    module=__name__,
                    request_id=f"claim-{round_id}",
                    round_id=round_id,
                    context={"error": str(e)},
                )
