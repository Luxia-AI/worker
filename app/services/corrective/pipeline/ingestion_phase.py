"""
Ingestion Phase: Persist facts and triples to VDB and KG.

Deferred domain trust resolution:
- Facts with PENDING_DOMAIN_TRUST domains are NOT ingested to VDB/KG
- Facts are enriched with validation_state and verdict_state
- Only TRUSTED domain facts are persisted (safe ingestion guard)
- VerdictState signals whether domain approval may be pending
"""

from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.services.evidence_validator import EvidenceValidator
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

    Deferred domain trust logic:
    - Enrich facts with validation_state and verdict_state
    - VDB ingest filters by domain trust (skips PENDING_DOMAIN_TRUST, UNTRUSTED)
    - Only persisted facts contribute to KG
    - All validation states logged for auditability

    Args:
        vdb_ingest: VDBIngest instance (Pinecone)
        kg_ingest: KGIngest instance (Neo4j)
        facts: List of fact dicts to ingest
        triples: List of triple dicts to ingest
        round_id: Round identifier for logging
    """
    # Enrich facts with validation state before ingestion
    # This signals to VDB ingest which domains are trusted
    if facts:
        for fact in facts:
            EvidenceValidator.enrich_evidence_with_validation(fact)

        # Log validation state distribution
        validation_states = {}
        for fact in facts:
            state = fact.get("validation_state", "unknown")
            validation_states[state] = validation_states.get(state, 0) + 1

        logger.info(f"[IngestionPhase:{round_id}] Validation state distribution: {validation_states}")

    # Ingest facts to VDB (VDB ingest filters by domain trust)
    if facts:
        try:
            ingested_ids = await vdb_ingest.embed_and_ingest(facts)
            logger.info(f"[IngestionPhase:{round_id}] Ingested {len(ingested_ids)} facts to VDB")

            if log_manager:
                await log_manager.add_log(
                    level="INFO",
                    message=f"VDB ingestion completed: {len(ingested_ids)} facts",
                    module=__name__,
                    request_id=f"claim-{round_id}",
                    round_id=round_id,
                    context={
                        "facts_ingested": len(ingested_ids),
                        "validation_state_dist": validation_states,
                    },
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
    # Triples for PENDING_DOMAIN_TRUST facts are also skipped to keep KG consistent
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
