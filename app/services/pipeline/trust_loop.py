"""
Iterative Trust Resolution Pipeline

This module orchestrates the iterative trust loop for Hybrid RAG systems,
integrating evidence retrieval, ranking, and corrective actions.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.ranking.trust_ranker import (
    MAX_EVIDENCE_POOL,
    MAX_ITERATIONS_DEFAULT,
    MIN_TRUST_IMPROVEMENT,
    STOP_NO_IMPROVEMENT,
    STOP_NO_NEW_EVIDENCE,
    STOP_THRESHOLD_MET,
    TRIGGER_CORRECTIVE,
    TRUST_THRESHOLD,
    EvidenceItem,
    IterationState,
    TrustRankingModule,
)

logger = get_logger(__name__)


class IterativeTrustResolver:
    """
    Deterministic iterative trust loop for Hybrid RAG system.

    Implements the full trust resolution pipeline with corrective retrieval.
    """

    def __init__(
        self,
        trust_module: TrustRankingModule,
        max_iterations: int = MAX_ITERATIONS_DEFAULT,
        trust_threshold: float = TRUST_THRESHOLD,
        min_trust_improvement: float = MIN_TRUST_IMPROVEMENT,
    ):
        self.trust_module = trust_module
        self.max_iterations = max_iterations
        self.trust_threshold = trust_threshold
        self.min_trust_improvement = min_trust_improvement

    async def run(self, claim: str) -> Dict[str, Any]:
        """
        Execute the iterative trust resolution loop.

        Args:
            claim: The health claim to evaluate

        Returns:
            Dict with response, trust_score, explanation, and trust_history
        """
        iteration_count = 0
        previous_trust_post = 0.0
        trust_history: List[IterationState] = []
        evidence_pool: List[EvidenceItem] = []

        while iteration_count < self.max_iterations:
            iteration_count += 1
            logger.info(f"[IterativeTrustResolver] Starting iteration {iteration_count}")

            # Phase 1: Embed claim
            claim_embedding = self.embed_claim(claim)

            # Phase 2: Retrieve internal evidence (VDB + KG)
            new_evidence = await self.retrieve_internal_evidence(claim, claim_embedding)
            new_evidence_count = len(new_evidence)
            evidence_pool.extend(new_evidence)

            # Phase 3: Classify stance (MUST run before ranking)
            self.classify_stance(claim, evidence_pool)

            # Phase 4: Rank evidence
            ranked_evidence = self.rank_evidence(evidence_pool)

            # Truncate evidence pool for stability
            evidence_pool = ranked_evidence[:MAX_EVIDENCE_POOL]

            # Phase 5: Compute post-level trust (single computation on truncated pool)
            post_trust_result = self.compute_post_trust(evidence_pool)
            current_trust_post = post_trust_result["trust_post"]
            trust_delta = current_trust_post - previous_trust_post

            # Decide next action
            decision = self.decide_next_action(iteration_count, current_trust_post, trust_delta)

            # Determine stop reason
            stop_reason = None
            if current_trust_post >= self.trust_threshold:
                stop_reason = STOP_THRESHOLD_MET
            elif iteration_count > 1 and trust_delta < self.min_trust_improvement:
                stop_reason = STOP_NO_IMPROVEMENT

            # Record iteration state
            top_sources = [item.source_url for item in evidence_pool[:5] if item.source_url]

            iteration_state = IterationState(
                iteration=iteration_count,
                trust_post=current_trust_post,
                agreement_ratio=post_trust_result["agreement_ratio"],
                top_sources=top_sources,
                decision=decision,
                trust_delta=trust_delta,
                evidence_count=len(evidence_pool),
                stop_reason=stop_reason,
                new_evidence_count=new_evidence_count,
            )
            trust_history.append(iteration_state)

            logger.info(
                f"[IterativeTrustResolver] Iteration {iteration_count}: "
                f"trust_post={current_trust_post:.3f}, delta={trust_delta:.3f}, "
                f"decision={decision}, evidence={len(evidence_pool)}, new={new_evidence_count}"
            )

            # Check stop conditions
            if stop_reason:
                logger.info(
                    f"[IterativeTrustResolver] Stopping: {stop_reason} "
                    f"(trust_post={current_trust_post:.3f}, delta={trust_delta:.3f})"
                )
                break

            # Phase 6: Decide next action (if corrective needed)
            if decision == TRIGGER_CORRECTIVE:
                # Phase 7: Corrective web retrieval
                corrective_evidence = self.corrective_web_retrieval(claim)
                corrective_evidence = self.dedupe_evidence(corrective_evidence)
                if not corrective_evidence:
                    stop_reason = STOP_NO_NEW_EVIDENCE
                    logger.info(
                        f"[IterativeTrustResolver] Stopping: {stop_reason} "
                        f"(no new evidence from corrective retrieval)"
                    )
                    # Update last state with stop_reason set after corrective failure
                    trust_history[-1].stop_reason = stop_reason
                    break
                # Phase 8: Enrich stores (VDB + KG)
                enriched_count = self.enrich_stores(corrective_evidence)
                if enriched_count == 0:
                    stop_reason = STOP_NO_NEW_EVIDENCE
                    logger.info(f"[IterativeTrustResolver] Stopping: {stop_reason} (no new evidence enriched)")
                    trust_history[-1].stop_reason = stop_reason
                    break

            previous_trust_post = current_trust_post

        # Phase 9: Generate final response
        return self.generate_final_response(claim, trust_history, evidence_pool)

    def embed_claim(self, claim: str) -> Any:
        """Embed the claim for retrieval. TODO: Integrate with embedding service."""
        # Placeholder: return None, implement actual embedding
        logger.debug(f"[IterativeTrustResolver] Embedding claim: {claim[:50]}...")
        return None

    async def retrieve_internal_evidence(self, claim: str, claim_embedding: Any) -> List[EvidenceItem]:
        """
        Retrieve evidence from Vector DB and Knowledge Graph.

        KG facts must be normalized to textual statements and embedded identically to VDB evidence.
        """
        evidence_items = []

        # TODO: Retrieve from Vector DB
        # vdb_results = self.retrieve_from_vdb(claim_embedding)
        # for result in vdb_results:
        #     evidence_items.append(EvidenceItem(...))

        # Retrieve KG triples and convert to EvidenceItem
        try:
            from app.services.retrieval.kg_normalizer import triples_to_evidence
            from app.services.retrieval.neo4j_kg_repository import Neo4jKGRepository

            kg_repo = Neo4jKGRepository()

            # TODO: Implement proper semantic_score_provider using embedding similarity
            # For now, use placeholder scoring
            def semantic_score_provider(statement: str) -> float:
                # Placeholder: return fixed score, in production use embedding similarity
                return 0.5

            kg_triples = await kg_repo.fetch_triples_for_claim(entity_ids=self.extract_entity_ids(claim), limit=100)
            if kg_triples:
                kg_evidence = triples_to_evidence(kg_triples, semantic_score_provider)
                evidence_items.extend(kg_evidence)
        except Exception as e:
            logger.warning("[IterativeTrustResolver] KG retrieval failed: %s", str(e))
            # Continue without KG evidence

        logger.debug(f"[IterativeTrustResolver] Retrieved {len(evidence_items)} internal evidence items")
        return evidence_items

    def extract_entity_names(self, claim: str) -> List[str]:
        """
        Extract potential entity names from a claim for KG anchoring.

        This is a simple heuristic - in production, use proper NER.
        """
        import re

        # Simple entity extraction: capitalized words, medical terms
        words = re.findall(r"\b[A-Z][a-z]+\b", claim)

        # Add common medical/health entities that might be mentioned
        medical_terms = ["COVID", "vaccine", "virus", "disease", "treatment", "symptoms"]
        entities = list(set(words + medical_terms))

        # Filter to reasonable length entities
        entities = [e for e in entities if 2 <= len(e) <= 50]

        logger.debug(f"[IterativeTrustResolver] Extracted entities from claim: {entities}")
        return entities

    def extract_entity_ids(self, claim: str) -> List[str]:
        """
        Extract entity IDs from a claim using the same deterministic generation as KG ingestion.
        """
        import hashlib

        entity_names = self.extract_entity_names(claim)
        entity_ids = []

        for name in entity_names:
            # Use same logic as kg_ingest._generate_entity_id
            normalized = name.lower().strip()
            entity_id = hashlib.md5(normalized.encode()).hexdigest()[:16]
            entity_ids.append(entity_id)

        logger.debug(f"[IterativeTrustResolver] Generated entity IDs: {entity_ids}")
        return entity_ids

    def classify_stance(self, claim: str, evidence_list: List[EvidenceItem]) -> None:
        """Classify stance for all evidence items."""
        self.trust_module.classify_stance_for_evidence(claim, evidence_list)

    def rank_evidence(self, evidence_list: List[EvidenceItem]) -> List[EvidenceItem]:
        """Rank evidence using TrustRankingModule."""
        return self.trust_module.rank_evidence(evidence_list)

    def compute_post_trust(self, ranked_evidence: List[EvidenceItem]) -> Dict[str, Any]:
        """Compute post-level trust metrics."""
        return self.trust_module.compute_post_trust(ranked_evidence)

    def decide_next_action(self, iteration: int, trust_post: float, trust_delta: float) -> str:
        """Decide whether to continue or stop."""
        if trust_post >= self.trust_threshold:
            return STOP_THRESHOLD_MET
        elif iteration == 1:
            return TRIGGER_CORRECTIVE
        elif trust_delta < self.min_trust_improvement:
            return STOP_NO_IMPROVEMENT
        else:
            return TRIGGER_CORRECTIVE

    def dedupe_evidence(self, evidence_list: List[EvidenceItem]) -> List[EvidenceItem]:
        """Deduplicate evidence by normalized source_url or md5(statement.lower().strip())."""
        seen = set()
        deduped = []
        for item in evidence_list:
            if item.source_url:
                # Use consistent URL normalization from TrustRankingModule
                key = self.trust_module._normalize_url_for_dedupe(item.source_url)
            else:
                key = hashlib.md5(item.statement.lower().strip().encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def enrich_stores(self, evidence: List[EvidenceItem]) -> int:
        """Enrich Vector DB and Knowledge Graph with new evidence.
        TODO: Implement upsert. Returns count of enriched items."""
        logger.debug(f"[IterativeTrustResolver] Enriching stores with {len(evidence)} new evidence items")
        # TODO: Implement actual enrichment, return actual count
        return len(evidence)  # Placeholder: assume all enriched

    def generate_final_response(
        self, claim: str, trust_history: List[IterationState], ranked_evidence: List[EvidenceItem]
    ) -> Dict[str, Any]:
        """
        Generate final response with hard gate on trust threshold.

        Blocks generation unless trust_post >= TRUST_THRESHOLD.
        """
        final_trust = trust_history[-1].trust_post if trust_history else 0.0

        if final_trust >= self.trust_threshold:
            # TODO: Implement actual response generation based on evidence
            response = f"Based on reliable evidence, the claim '{claim}' is evaluated."  # Placeholder
            return {
                "response": response,
                "trust_score": final_trust,
                "iterations": len(trust_history),
                "trust_history": [vars(state) for state in trust_history],
            }
        else:
            return {
                "response": None,
                "trust_score": final_trust,
                "iterations": len(trust_history),
                "explanation": f"Insufficient reliable evidence after {len(trust_history)} iterations",
                "trust_history": [vars(state) for state in trust_history],
            }
