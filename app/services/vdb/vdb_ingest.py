from typing import Any, Dict, List, Set

from app.constants.config import ValidationState
from app.core.logger import get_logger
from app.services.embedding.model import embed_async
from app.services.evidence_validator import EvidenceValidator
from app.services.retrieval.lexical_index import LexicalIndex
from app.services.retrieval.metadata_enricher import MetadataEnricher
from app.services.vdb.pinecone_client import get_pinecone_index

logger = get_logger(__name__)


class VDBIngest:
    def __init__(self, namespace: str = "health"):
        self.index = get_pinecone_index()
        self.namespace = namespace
        self.metadata_enricher = MetadataEnricher()
        self.lexical_index = LexicalIndex()

    def get_processed_urls(
        self,
        topics: List[str] | None = None,
        max_urls: int | None = None,
    ) -> Set[str]:
        """
        Get set of URLs that have already been processed and ingested to VDB.

        Uses Pinecone's list operation to get vector IDs and extracts unique source URLs.

        Args:
            topics: Optional list of claim topics. When provided, only URLs from matching
                topic metadata are returned (plus "other" as permissive fallback).
            max_urls: Optional hard cap for number of collected URLs.

        Returns:
            Set of source URLs that exist in the VDB namespace
        """
        try:
            # Get all vector IDs in namespace using pagination
            processed_urls: Set[str] = set()
            topic_filter = {t.strip().lower() for t in (topics or []) if t and t.strip()}
            if "other" not in topic_filter and topic_filter:
                topic_filter.add("other")

            # Pinecone's list returns an iterator
            for ids_batch in self.index.list(namespace=self.namespace):
                if not ids_batch:
                    continue

                # Fetch metadata for these IDs
                try:
                    fetch_response = self.index.fetch(ids=ids_batch, namespace=self.namespace)
                    if hasattr(fetch_response, "vectors"):
                        vectors = fetch_response.vectors or {}
                    elif isinstance(fetch_response, dict):
                        vectors = fetch_response.get("vectors", {})
                    else:
                        vectors = {}

                    for vec_id, vec_data in vectors.items():
                        metadata = vec_data.get("metadata", {})
                        topic = str(metadata.get("topic") or "").strip().lower()
                        if topic_filter and topic and topic not in topic_filter:
                            continue
                        if topic_filter and not topic:
                            continue
                        source_url = metadata.get("source_url", "")
                        if source_url:
                            processed_urls.add(source_url)
                        if max_urls is not None and len(processed_urls) >= max(1, int(max_urls)):
                            logger.info(
                                "[VDBIngest] Reached processed URL cap (%d) for namespace=%s",
                                len(processed_urls),
                                self.namespace,
                            )
                            return processed_urls
                except Exception as e:
                    logger.warning(f"[VDBIngest] Failed to fetch metadata for batch: {e}")
                    continue

            logger.info(
                "[VDBIngest] Found %d unique processed URLs in VDB (namespace=%s, topic_filter=%s)",
                len(processed_urls),
                self.namespace,
                sorted(topic_filter) if topic_filter else "none",
            )
            return processed_urls

        except Exception as e:
            logger.warning(f"[VDBIngest] Failed to get processed URLs: {e}")
            return set()

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
                f"[VDBIngest] Filtered {len(facts)} facts -> {len(trusted_facts)} trusted, {skipped_count} skipped"
            )

        # Enrich trusted facts with metadata before embedding
        trusted_facts = await self.metadata_enricher.enrich_facts(trusted_facts)

        logger.info(f"[VDBIngest] Embedding {len(trusted_facts)} facts from trusted domains")

        statements = [f["statement"] for f in trusted_facts]
        embeddings = await embed_async(statements)

        vectors = []
        for fact, emb in zip(trusted_facts, embeddings):
            claim_ctx_entities = fact.get("claim_context_entities") or fact.get("claim_entities_ctx") or []
            if isinstance(claim_ctx_entities, str):
                claim_ctx_entities = [claim_ctx_entities]
            claim_ctx_entities = [str(e).strip().lower() for e in claim_ctx_entities if str(e).strip()][:20]
            metadata = {
                "statement": fact["statement"],
                "entities": fact.get("entities", []),
                "source_url": fact.get("source_url"),
                "language": fact.get("language", "en"),
                "domain": fact.get("domain"),
                "topic": fact.get("topic"),
                "source": fact.get("source"),
                "doc_type": fact.get("doc_type"),
                "fact_type": fact.get("fact_type"),
                "count_value": fact.get("count_value"),
                "claim_context_hash": str(fact.get("claim_context_hash") or "").strip() or None,
                "claim_context_entities": claim_ctx_entities or None,
            }
            # Pinecone metadata cannot include null values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            vectors.append(
                {
                    "id": fact["fact_id"],
                    "values": emb,
                    "metadata": metadata,
                }
            )

        logger.info(f"[VDBIngest] Upserting into Pinecone: {len(vectors)} vectors")

        self.index.upsert(vectors=vectors, namespace=self.namespace)

        # Update lexical BM25 index (best-effort)
        try:
            self.lexical_index.upsert_facts(trusted_facts)
        except Exception as e:
            logger.warning(f"[VDBIngest] Lexical index update failed: {e}")

        return [v["id"] for v in vectors]
