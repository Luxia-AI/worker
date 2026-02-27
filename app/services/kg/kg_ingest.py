from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.kg.neo4j_client import NEO4J_QUERY_TIMEOUT, Neo4jClient

logger = get_logger(__name__)

# Overall timeout for the entire ingest operation
KG_INGEST_TIMEOUT = 30  # seconds


class KGIngest:
    """
    Handles ingestion of entity-relation-entity triples into Neo4j.

    Creates proper KG structure:
        (Entity {id, name}) -[:SUBJECT_OF]-> (Relation {rid, predicate, confidence, updated_at})
        -[:OBJECT_OF]-> (Entity {id, name})
        (Relation) -[:SUPPORTED_BY]-> (Source {url, domain})

    Ensures:
        - Entity.id is stable and deterministic
        - Relation.rid is deterministic: "{subject_id}|{predicate}|{object_id}"
        - Relations are unique by rid
        - Confidence and timestamps use SET, not MERGE
        - Logs every action
        - Fails gracefully with timeout if Neo4j is unavailable
    """

    def __init__(self) -> None:
        self.client = Neo4jClient()

    @staticmethod
    def _normalize_relation(relation: str) -> str:
        return re.sub(r"\s+", " ", (relation or "").replace("_", " ").strip().lower())

    @staticmethod
    def _is_negated_statement(statement: str) -> bool:
        text = (statement or "").lower()
        negation_patterns = [
            r"\b(?:do|does|did|is|are|was|were|can|could|may|might|must|should|would|will)\s+not\b",
            r"\bno\s+(?:evidence|link|association|causal link)\b",
            r"\bnot\s+(?:associated|linked|causal)\b",
            r"\bunrelated to\b",
            r"\b(?:myth|debunked|false claim|no link)\b",
        ]
        return any(re.search(pat, text) for pat in negation_patterns)

    def _should_skip_triple(self, triple: Dict[str, Any]) -> bool:
        relation = self._normalize_relation(triple.get("relation", ""))
        if not relation:
            return True
        if relation.startswith(("does_not_", "not_")):
            return False
        causal_markers = (
            "cause",
            "contribut",
            "associat",
            "link",
            "increase risk",
            "lead to",
            "result in",
        )
        if not any(marker in relation for marker in causal_markers):
            return False
        statement = str(triple.get("source_statement", "") or "")
        return bool(statement and self._is_negated_statement(statement))

    async def ingest_triples(self, triples: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Ingest triples into Neo4j KG.

        Returns:
            Dict with counts: {"attempted": int, "succeeded": int, "failed": int}
        """
        if not triples:
            return {"attempted": 0, "succeeded": 0, "failed": 0}

        # Wrap entire operation in timeout to prevent blocking
        try:
            return await asyncio.wait_for(self._do_ingest(triples), timeout=KG_INGEST_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"[KGIngest] Overall timeout ({KG_INGEST_TIMEOUT}s), skipping KG ingestion")
            return {"attempted": len(triples), "succeeded": 0, "failed": len(triples)}
        except ConnectionError as e:
            logger.warning(f"[KGIngest] Neo4j unavailable, skipping: {e}")
            return {"attempted": len(triples), "succeeded": 0, "failed": len(triples)}
        except Exception as e:
            logger.error(f"[KGIngest] Unexpected error, skipping: {e}")
            return {"attempted": len(triples), "succeeded": 0, "failed": len(triples)}

    async def _do_ingest(self, triples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Internal method that performs the actual ingestion."""
        attempted = len(triples)
        succeeded = 0
        failed = 0

        try:
            async with self.client.session() as session:
                for triple in triples:
                    if self._should_skip_triple(triple):
                        logger.warning(f"[KGIngest] Skipped negated/unsafe triple: {triple}")
                        failed += 1
                        continue
                    subj_name = triple.get("subject", "").strip()
                    rel_predicate = triple.get("relation", "").strip()
                    obj_name = triple.get("object", "").strip()

                    if not subj_name or not rel_predicate or not obj_name:
                        logger.warning(f"[KGIngest] Skipped malformed triple: {triple}")
                        failed += 1
                        continue

                    # Generate deterministic IDs
                    subj_id = self._generate_entity_id(subj_name)
                    obj_id = self._generate_entity_id(obj_name)
                    rel_rid = self._generate_relation_rid(subj_id, rel_predicate, obj_id)

                    # Extract optional fields
                    confidence = float(triple.get("confidence", 0.0))
                    source_url = triple.get("source_url")
                    claim_context_hash = str(triple.get("claim_context_hash") or "").strip()
                    claim_context_entities = (
                        triple.get("claim_context_entities") or triple.get("claim_entities_ctx") or []
                    )
                    if isinstance(claim_context_entities, str):
                        claim_context_entities = [claim_context_entities]
                    claim_context_entities = [str(e).strip().lower() for e in claim_context_entities if str(e).strip()][
                        :20
                    ]

                    # Ingest the triple
                    try:
                        await asyncio.wait_for(
                            self._ingest_single_triple(
                                session,
                                subj_id,
                                subj_name,
                                obj_id,
                                obj_name,
                                rel_rid,
                                rel_predicate,
                                confidence,
                                source_url,
                                claim_context_hash,
                                claim_context_entities,
                            ),
                            timeout=NEO4J_QUERY_TIMEOUT,
                        )
                        succeeded += 1
                    except asyncio.TimeoutError:
                        logger.warning(f"[KGIngest] Query timeout for triple: {subj_name}->{rel_predicate}->{obj_name}")
                        failed += 1
                    except Exception as e:
                        logger.error(f"[KGIngest] Failed to ingest triple {rel_rid}: {e}")
                        failed += 1

        except ConnectionError as e:
            logger.warning(f"[KGIngest] Connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[KGIngest] Session failed: {e}")
            raise

        logger.info(f"[KGIngest] Completed ingestion: {attempted} attempted, {succeeded} succeeded, {failed} failed")
        return {"attempted": attempted, "succeeded": succeeded, "failed": failed}

    def _generate_entity_id(self, entity_name: str) -> str:
        """Generate deterministic Entity.id from entity name."""
        # Use hash of normalized name for stability
        normalized = entity_name.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _generate_relation_rid(self, subject_id: str, predicate: str, object_id: str) -> str:
        """Generate deterministic Relation.rid."""
        return f"{subject_id}|{predicate}|{object_id}"

    async def _ingest_single_triple(
        self,
        session,
        subj_id: str,
        subj_name: str,
        obj_id: str,
        obj_name: str,
        rel_rid: str,
        rel_predicate: str,
        confidence: float,
        source_url: str | None,
        claim_context_hash: str = "",
        claim_context_entities: List[str] | None = None,
    ) -> None:
        """Ingest a single triple with proper KG structure."""
        claim_context_entities = claim_context_entities or []

        # Cypher query for proper KG structure
        cypher = """
        // Merge entities by id (stable identifier)
        MERGE (subj:Entity {id: $subj_id})
        ON CREATE SET subj.name = $subj_name
        ON MATCH SET subj.name = CASE WHEN subj.name <> $subj_name THEN $subj_name ELSE subj.name END

        MERGE (obj:Entity {id: $obj_id})
        ON CREATE SET obj.name = $obj_name
        ON MATCH SET obj.name = CASE WHEN obj.name <> $obj_name THEN $obj_name ELSE obj.name END

        // Merge relation by rid (deterministic identifier)
        MERGE (rel:Relation {rid: $rel_rid})
        ON CREATE SET
            rel.predicate = $rel_predicate,
            rel.confidence = $confidence,
            rel.claim_context_hash = CASE WHEN $claim_context_hash <> '' THEN $claim_context_hash ELSE NULL END,
            rel.claim_context_entities = CASE
                WHEN size($claim_context_entities) > 0 THEN $claim_context_entities
                ELSE NULL
            END,
            rel.updated_at = datetime()
        ON MATCH SET
            rel.confidence = CASE WHEN rel.confidence < $confidence THEN $confidence ELSE rel.confidence END,
            rel.claim_context_hash = CASE
                WHEN $claim_context_hash <> '' THEN $claim_context_hash
                ELSE rel.claim_context_hash
            END,
            rel.claim_context_entities = CASE
                WHEN size($claim_context_entities) > 0 THEN $claim_context_entities
                ELSE rel.claim_context_entities
            END,
            rel.updated_at = datetime()

        // Create relationships (idempotent)
        MERGE (subj)-[:SUBJECT_OF]->(rel)
        MERGE (rel)-[:OBJECT_OF]->(obj)
        """

        params = {
            "subj_id": subj_id,
            "subj_name": subj_name,
            "obj_id": obj_id,
            "obj_name": obj_name,
            "rel_rid": rel_rid,
            "rel_predicate": rel_predicate,
            "confidence": confidence,
            "claim_context_hash": claim_context_hash,
            "claim_context_entities": claim_context_entities,
        }

        await session.run(cypher, **params)

        # Handle source if provided
        if source_url:
            await self._ingest_source(session, rel_rid, source_url)

    async def _ingest_source(self, session, rel_rid: str, source_url: str) -> None:
        """Ingest source and link to relation."""
        # Extract domain from URL
        domain = self._extract_domain(source_url)

        cypher = """
        // Merge source by url
        MERGE (src:Source {url: $source_url})
        ON CREATE SET src.domain = $domain

        // Merge relation by rid (ensure it exists)
        MERGE (rel:Relation {rid: $rel_rid})

        // Create relationship (idempotent)
        MERGE (rel)-[:SUPPORTED_BY]->(src)
        """

        params = {
            "rel_rid": rel_rid,
            "source_url": source_url,
            "domain": domain,
        }

        await session.run(cypher, **params)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for Source.domain property."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"
