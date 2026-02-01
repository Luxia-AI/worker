from typing import Any, Dict, List

from app.constants.config import ValidationState
from app.core.logger import get_logger
from app.services.embedding.model import embed_async
from app.services.evidence_validator import EvidenceValidator
from app.services.vdb.pinecone_client import get_pinecone_index

logger = get_logger(__name__)


class VDBIngest:
    def __init__(self, namespace: str = "health"):
        self.index = get_pinecone_index()
        self.namespace = namespace

    async def embed_and_ingest(self, facts: List[Dict[str, Any]]) -> List[str]:
        """
        Embed and ingest facts to Pinecone, but ONLY if domain is trusted.

        Non-blocking persistence:
        - Facts with PENDING_DOMAIN_TRUST or UNTRUSTED domains are skipped
        - Only TRUSTED domain facts are persisted
        - Failures in embedding/upserting are logged but don't block pipeline

        Args:
            facts: List of fact dicts with 'source_url' and 'statement' keys

        Returns:
            List of ingested fact IDs (excludes untrusted domain facts)
        """
        if not facts:
            return []

        # Filter to only trusted domains (safe ingestion guard)
        # PENDING_DOMAIN_TRUST facts are NOT ingested until domain is approved
        trusted_facts = []
        skipped_count = 0

        for fact in facts:
            source_url = fact.get("source_url", "")
            validation_state = EvidenceValidator.get_validation_state(source_url)

            if validation_state == ValidationState.TRUSTED:
                trusted_facts.append(fact)
            else:
                # Skip ingestion for untrusted/pending domains
                skipped_count += 1
                reason = "untrusted" if validation_state == ValidationState.UNTRUSTED else "pending_domain_trust"
                logger.info(f"[VDBIngest] Skipping ingestion for {reason} domain: {source_url}")

        if not trusted_facts:
            logger.info(f"[VDBIngest] All {len(facts)} facts skipped (no trusted domains)")
            return []

        if skipped_count > 0:
            logger.info(
                f"[VDBIngest] Filtered {len(facts)} facts → {len(trusted_facts)} trusted, {skipped_count} skipped"
            )

        logger.info(f"[VDBIngest] Embedding {len(trusted_facts)} facts from trusted domains")

        statements = [f["statement"] for f in trusted_facts]
        embeddings = await embed_async(statements)

        vectors = []
        for fact, emb in zip(trusted_facts, embeddings):
            vectors.append(
                {
                    "id": fact["fact_id"],
                    "values": emb,
                    "metadata": {
                        "statement": fact["statement"],
                        "entities": fact.get("entities", []),
                        "source_url": fact.get("source_url"),
                    },
                }
            )

        logger.info(f"[VDBIngest] Upserting into Pinecone: {len(vectors)} vectors")

        self.index.upsert(vectors=vectors, namespace=self.namespace)

        return [v["id"] for v in vectors]
