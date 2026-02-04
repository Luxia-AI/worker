"""
Neo4j Knowledge Graph Repository for triple retrieval.

Provides clean access to KG triples for trust ranking integration.
"""

from __future__ import annotations

import asyncio
import time
from typing import List

from app.core.logger import get_logger
from app.services.kg.neo4j_client import Neo4jClient
from app.services.retrieval.kg_normalizer import KGTriple

logger = get_logger(__name__)

# Query timeout for KG retrieval
KG_QUERY_TIMEOUT = 10  # seconds

# Retry configuration for Neo4j Aura Free
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 8.0

# Transient error codes that should trigger retry
TRANSIENT_ERROR_CODES = {
    "Neo.TransientError.General.OutOfMemoryError",
    "Neo.TransientError.General.DatabaseUnavailable",
    "Neo.TransientError.Network.CommunicationError",
    "Neo.TransientError.Transaction.DeadlockDetected",
    "Neo.TransientError.Transaction.LockClientStopped",
}


def _is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried."""
    error_str = str(error)
    return any(code in error_str for code in TRANSIENT_ERROR_CODES)


def _calculate_backoff_delay(retry_count: int) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(BASE_BACKOFF_SECONDS * (2**retry_count), MAX_BACKOFF_SECONDS)
    # Add jitter (±25%)
    jitter = delay * 0.25 * (0.5 - time.time() % 1)  # Pseudo-random jitter
    return max(0.1, delay + jitter)


async def _execute_with_retry(
    query: str, params: dict, session, query_name: str, timeout: float = KG_QUERY_TIMEOUT
) -> List[dict]:
    """
    Execute a Cypher query with retry logic and structured logging.

    Args:
        query: Cypher query string
        params: Query parameters
        session: Neo4j session
        query_name: Name for logging
        timeout: Query timeout in seconds

    Returns:
        List of result records

    Raises:
        Exception: Final error after all retries exhausted
    """
    last_error = None

    for retry_count in range(MAX_RETRIES + 1):  # +1 for initial attempt
        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(session.run(query, params), timeout=timeout)
            records = await result.data()

            # Log successful execution
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[Neo4jKGRepository] Query '{query_name}' completed",
                extra={
                    "query_name": query_name,
                    "duration_ms": duration_ms,
                    "returned_rows": len(records),
                    "retry_count": retry_count,
                    "success": True,
                },
            )

            return records

        except asyncio.TimeoutError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(
                f"[Neo4jKGRepository] Query '{query_name}' timed out after "
                f"{duration_ms}ms (attempt {retry_count + 1}/{MAX_RETRIES + 1})",
                extra={
                    "query_name": query_name,
                    "duration_ms": duration_ms,
                    "retry_count": retry_count,
                    "error_type": "timeout",
                    "success": False,
                },
            )
            last_error = e

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            is_transient = _is_transient_error(e)

            logger.warning(
                f"[Neo4jKGRepository] Query '{query_name}' failed after "
                f"{duration_ms}ms (attempt {retry_count + 1}/{MAX_RETRIES + 1}): {str(e)}",
                extra={
                    "query_name": query_name,
                    "duration_ms": duration_ms,
                    "retry_count": retry_count,
                    "error_type": "transient" if is_transient else "permanent",
                    "success": False,
                },
            )
            last_error = e

            # Don't retry permanent errors
            if not is_transient:
                break

        # Don't backoff on last attempt
        if retry_count < MAX_RETRIES:
            delay = _calculate_backoff_delay(retry_count)
            logger.info(
                f"[Neo4jKGRepository] Retrying query '{query_name}' in "
                f"{delay:.2f}s (attempt {retry_count + 2}/{MAX_RETRIES + 1})",
                extra={"query_name": query_name, "retry_count": retry_count + 1, "backoff_delay_seconds": delay},
            )
            await asyncio.sleep(delay)

    # All retries exhausted
    logger.error(
        f"[Neo4jKGRepository] Query '{query_name}' failed after {MAX_RETRIES + 1} attempts",
        extra={"query_name": query_name, "retry_count": MAX_RETRIES, "final_error": str(last_error), "success": False},
    )
    raise last_error


class Neo4jKGRepository:
    """
    Repository for retrieving KG triples from Neo4j Aura.

    Provides deterministic, safe access to knowledge graph data for trust ranking.
    """

    def __init__(self, neo4j_client: Neo4jClient | None = None):
        self.neo4j_client = neo4j_client or Neo4jClient()

    async def fetch_triples_for_claim(
        self, claim_id: str | None = None, entity_ids: List[str] | None = None, limit: int = 100
    ) -> List[KGTriple]:
        """
        Fetch KG triples relevant to a claim or entity IDs.

        Uses anchored Cypher query to retrieve claim-specific triples.
        Either claim_id or entity_ids must be provided.

        Args:
            claim_id: ID of the claim to fetch triples for (optional)
            entity_ids: List of entity IDs to anchor the search (optional)
            limit: Maximum triples to return (default 100, max 1000)

        Returns:
            List of KGTriple objects with normalized data
        """
        if not claim_id and not entity_ids:
            raise ValueError("Either claim_id or entity_ids must be provided")

        # Enforce reasonable limits for Neo4j Aura Free
        limit = min(max(1, limit), 1000)  # 1-1000 range

        # Build query based on parameters
        if claim_id:
            # Query anchored to specific claim
            query = """
            MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e1:Entity)
            MATCH (e1)-[:SUBJECT_OF]->(r:Relation)-[:OBJECT_OF]->(e2:Entity)
            OPTIONAL MATCH (r)-[:SUPPORTED_BY]->(s:Source)
            RETURN
                e1.name AS subject,
                e1.id AS subject_id,
                r.predicate AS predicate,
                r.rid AS rid,
                e2.name AS object,
                e2.id AS object_id,
                coalesce(r.confidence, 0.0) AS confidence,
                s.url AS source_url,
                s.domain AS source_domain,
                toString(r.updated_at) AS updated_at
            ORDER BY confidence DESC, updated_at DESC
            LIMIT $limit
            """
            params = {"claim_id": claim_id, "limit": limit}
        else:
            # Query anchored to entity IDs - EXACT required pattern
            query = """
            MATCH (sub:Entity)-[:SUBJECT_OF]->(r:Relation)-[:OBJECT_OF]->(obj:Entity)
            WHERE sub.id IN $entity_ids OR obj.id IN $entity_ids
            OPTIONAL MATCH (r)-[:SUPPORTED_BY]->(s:Source)
            RETURN
                sub.name AS subject,
                sub.id AS subject_id,
                r.predicate AS predicate,
                r.rid AS rid,
                obj.name AS object,
                obj.id AS object_id,
                coalesce(r.confidence, 0.0) AS confidence,
                s.url AS source_url,
                s.domain AS source_domain,
                toString(r.updated_at) AS updated_at
            ORDER BY confidence DESC, updated_at DESC
            LIMIT $limit
            """
            params = {"entity_ids": entity_ids, "limit": limit}

        triples = []

        try:
            async with self.neo4j_client.session() as session:
                # Execute with retry logic and structured logging
                records = await _execute_with_retry(
                    query=query,
                    params=params,
                    session=session,
                    query_name="fetch_kg_triples_for_claim",
                    timeout=KG_QUERY_TIMEOUT,
                )

                # Safeguard: truncate if somehow more rows returned than limit
                if len(records) > limit:
                    logger.warning(
                        f"[Neo4jKGRepository] Query returned {len(records)} rows, truncating to limit {limit}",
                        extra={
                            "query_name": "fetch_kg_triples_for_claim",
                            "returned_rows": len(records),
                            "limit": limit,
                            "truncated": True,
                        },
                    )
                    records = records[:limit]

                # Process records with robust null handling
                for record in records:
                    # Robust null handling
                    subject = record.get("subject", "").strip() if record.get("subject") else ""
                    subject_id = record.get("subject_id", "").strip() if record.get("subject_id") else ""
                    relation = record.get("predicate", "").strip() if record.get("predicate") else ""
                    rid = record.get("rid", "").strip() if record.get("rid") else ""
                    object_ = record.get("object", "").strip() if record.get("object") else ""
                    object_id = record.get("object_id", "").strip() if record.get("object_id") else ""
                    source_url = record.get("source_url")
                    source_domain = record.get("source_domain")
                    confidence = record.get("confidence", 0.0)  # coalesce ensures this is never None
                    updated_at = record.get("updated_at")

                    # Skip incomplete triples
                    if not subject or not relation or not object_ or not subject_id or not object_id or not rid:
                        continue

                    # Create KGTriple with safe defaults
                    triple = KGTriple(
                        subject=subject,
                        subject_id=subject_id,
                        relation=relation,
                        rid=rid,
                        object=object_,
                        object_id=object_id,
                        source_url=source_url,
                        source_domain=source_domain,
                        published_at=updated_at,  # Now returns as string
                        confidence=float(confidence),  # Always a valid float due to coalesce
                    )
                    triples.append(triple)

        except asyncio.TimeoutError:
            logger.warning("[Neo4jKGRepository] Query timed out after all retries")
        except Exception as e:
            logger.error("[Neo4jKGRepository] Error fetching triples after retries: %s", str(e))
            # Return empty list on error to fail gracefully

        logger.info("[Neo4jKGRepository] Retrieved %d KG triples", len(triples))
        return triples

    async def store_evidence_metadata(
        self, claim_text: str, evidence_count: int, sources_used: List[str], processing_timestamp: str
    ) -> bool:
        """
        Store metadata about evidence processing (NOT trust aggregates).

        This stores operational metadata for monitoring and debugging,
        but never stores trust_post scores or derived aggregates.

        Args:
            claim_text: The claim being processed
            evidence_count: Number of evidence items found
            sources_used: List of source URLs used
            processing_timestamp: ISO timestamp of processing

        Returns:
            True if stored successfully, False otherwise
        """
        # Hash claim for deterministic key (never use claim_text directly in MERGE)
        import hashlib

        claim_hash = hashlib.sha256(claim_text.encode()).hexdigest()[:16]

        query = """
        MERGE (c:Claim {hash: $claim_hash})
        ON CREATE SET
            c.text = $claim_text,
            c.created_at = datetime($processing_timestamp),
            c.evidence_count = $evidence_count,
            c.sources_used = $sources_used,
            c.last_processed = datetime($processing_timestamp)
        ON MATCH SET
            c.evidence_count = $evidence_count,
            c.sources_used = $sources_used,
            c.last_processed = datetime($processing_timestamp)
        """

        try:
            async with self.neo4j_client.session() as session:
                await _execute_with_retry(
                    query=query,
                    params={
                        "claim_hash": claim_hash,
                        "claim_text": claim_text,
                        "evidence_count": evidence_count,
                        "sources_used": sources_used,
                        "processing_timestamp": processing_timestamp,
                    },
                    session=session,
                    query_name="store_evidence_metadata",
                    timeout=KG_QUERY_TIMEOUT,
                )
            return True

        except Exception as e:
            logger.error("[Neo4jKGRepository] Failed to store evidence metadata: %s", str(e))
            return False
