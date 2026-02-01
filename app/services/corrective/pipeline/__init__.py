"""
Corrective Retrieval Pipeline: Main orchestrator.

OPTIMIZED ARCHITECTURE (Retrieval-First):
    1. Embedding: Embed claim and extract entities
    2. Retrieval: Check VDB + KG for existing evidence FIRST
    3. Ranking: If trust score >= threshold, DONE (no web search needed)
    4. Corrective Search: Only if rank < threshold, search incrementally
    5. Extraction: Extract facts from search results (batched)
    6. Ingestion: Store new evidence to VDB + KG for future use
    7. Re-rank: Combine new + existing evidence, check threshold again

This approach minimizes LLM calls by:
    - Using cached evidence when available
    - Processing search results in small batches (3-5 pages)
    - Batching fact extraction to reduce per-page LLM calls
    - Early termination when threshold reached
"""

import uuid
from typing import Any, Dict, List, Optional

from app.constants.config import PIPELINE_CONF_THRESHOLD, PIPELINE_MAX_ROUNDS, PIPELINE_MIN_NEW_URLS
from app.core.logger import get_logger
from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.pipeline.extraction_phase import extract_all, scrape_pages
from app.services.corrective.pipeline.ingestion_phase import ingest_facts_and_triples
from app.services.corrective.pipeline.ranking_phase import rank_candidates
from app.services.corrective.pipeline.retrieval_phase import retrieve_candidates
from app.services.corrective.pipeline.search_phase import do_search
from app.services.corrective.relation_extractor import RelationExtractor
from app.services.corrective.scraper import Scraper
from app.services.corrective.trusted_search import TrustedSearch
from app.services.kg.kg_ingest import KGIngest
from app.services.kg.kg_retrieval import KGRetrieval
from app.services.kg.neo4j_client import Neo4jClient
from app.services.llms.hybrid_service import reset_groq_counter
from app.services.logging.log_handler import LogManagerHandler
from app.services.vdb.vdb_ingest import VDBIngest
from app.services.vdb.vdb_retrieval import VDBRetrieval

logger = get_logger(__name__)

# Batch size for incremental web search processing
SEARCH_BATCH_SIZE = 5


class CorrectivePipeline:
    """
    Corrective Retrieval Pipeline with integrated hybrid ranking.

    Orchestrates:
        1. Query reformulation (TrustedSearch -> LLM)
        2. Trusted Search on whitelisted domains
        3. Page scraping and content extraction
        4. Fact/Entity/Relation extraction (LLM-powered)
        5. Ingestion to VDB (Pinecone) and KG (Neo4j)
        6. Retrieval: semantic (VDB) + structural (KG)
        7. Hybrid ranking combining 5 scoring signals
        8. Reinforcement loop for low-confidence results
    """

    MAX_ROUNDS = PIPELINE_MAX_ROUNDS
    CONF_THRESHOLD = PIPELINE_CONF_THRESHOLD
    MIN_NEW_URLS = PIPELINE_MIN_NEW_URLS

    def __init__(self) -> None:
        # Search and scraping
        self.search_agent = TrustedSearch()
        self.scraper = Scraper()

        # Extraction
        self.fact_extractor = FactExtractor()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()

        # Storage and retrieval
        self.vdb_ingest = VDBIngest()  # Pinecone
        self.vdb_retriever = VDBRetrieval()
        self.kg_client = Neo4jClient()
        self.kg_ingest = KGIngest()
        self.kg_retriever = KGRetrieval()

        # Logging system
        self.log_manager = LogManagerHandler._log_manager

    async def run(
        self,
        post_text: str,
        domain: str,
        failed_entities: Optional[List[str]] = None,
        round_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute the OPTIMIZED corrective retrieval pipeline.

        RETRIEVAL-FIRST APPROACH:
        1. Extract entities from claim (single LLM call)
        2. Retrieve existing evidence from VDB + KG
        3. Rank existing evidence
        4. IF trust score >= threshold: DONE (no web search, no extra LLM calls!)
        5. ELSE: Search incrementally, process in batches until threshold reached

        This minimizes Groq API calls by using cached knowledge when available.
        """
        round_id = round_id or str(uuid.uuid4())
        failed_entities = failed_entities or []

        # Reset Groq call counter for this new request
        reset_groq_counter()

        logger.info(f"[CorrectivePipeline:{round_id}] Start for post: {post_text}")

        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Pipeline started (retrieval-first): {post_text[:100]}...",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={"domain": domain},
            )

        # ====================================================================
        # PHASE 1: Extract entities from claim (1 LLM call)
        # ====================================================================
        claim_entities = await self._extract_claim_entities(post_text, round_id)
        logger.info(f"[CorrectivePipeline:{round_id}] Claim entities: {claim_entities}")

        # ====================================================================
        # PHASE 2: Retrieve existing evidence from VDB + KG (NO LLM calls)
        # ====================================================================
        dedup_sem, kg_candidates = await retrieve_candidates(
            self.vdb_retriever,
            self.kg_retriever,
            [post_text],  # Use claim as query
            claim_entities,
            top_k * 2,  # Get more candidates for better ranking
            round_id,
            self.log_manager,
        )

        logger.info(
            f"[CorrectivePipeline:{round_id}] Retrieved {len(dedup_sem)} VDB + {len(kg_candidates)} KG candidates"
        )

        # ====================================================================
        # PHASE 3: Rank existing evidence (NO LLM calls)
        # ====================================================================
        top_ranked = []
        if dedup_sem or kg_candidates:
            top_ranked = await rank_candidates(
                dedup_sem, kg_candidates, claim_entities, top_k, round_id, self.log_manager
            )

        # Check if we have sufficient evidence (trust score >= threshold)
        top_score = top_ranked[0]["final_score"] if top_ranked else 0.0
        logger.info(f"[CorrectivePipeline:{round_id}] Initial ranking: top_score={top_score:.3f}")

        if top_score >= self.CONF_THRESHOLD:
            # ✅ EARLY EXIT: Existing evidence is sufficient, no web search needed!
            logger.info(
                f"[CorrectivePipeline:{round_id}] Trust threshold met ({top_score:.3f} >= {self.CONF_THRESHOLD}). "
                "Skipping web search - using cached evidence!"
            )
            return {
                "round_id": round_id,
                "status": "completed_from_cache",
                "facts": [],  # No new facts extracted
                "triples": [],
                "queries": [post_text],
                "semantic_candidates_count": len(dedup_sem),
                "kg_candidates_count": len(kg_candidates),
                "ranked": top_ranked,
                "used_web_search": False,
            }

        # ====================================================================
        # PHASE 4: Corrective Search (incremental, batched)
        # ====================================================================
        logger.info(
            f"[CorrectivePipeline:{round_id}] Trust score too low ({top_score:.3f}), triggering corrective search..."
        )

        # Get search URLs
        search_urls, queries = await do_search(
            self.search_agent, post_text, failed_entities, round_id, self.log_manager
        )

        if not search_urls:
            logger.warning(f"[CorrectivePipeline:{round_id}] No search results found")
            return {
                "round_id": round_id,
                "status": "no_search_results",
                "facts": [],
                "triples": [],
                "queries": queries,
                "semantic_candidates_count": len(dedup_sem),
                "kg_candidates_count": len(kg_candidates),
                "ranked": top_ranked,
                "used_web_search": True,
            }

        # Process search results in batches
        all_facts: List[Dict[str, Any]] = []
        all_triples: List[Dict[str, Any]] = []
        all_entities = list(claim_entities)
        processed_urls: List[str] = []

        for batch_idx in range(0, len(search_urls), SEARCH_BATCH_SIZE):
            batch_urls = search_urls[batch_idx : batch_idx + SEARCH_BATCH_SIZE]
            batch_num = batch_idx // SEARCH_BATCH_SIZE + 1
            total_batches = (len(search_urls) + SEARCH_BATCH_SIZE - 1) // SEARCH_BATCH_SIZE

            logger.info(
                f"[CorrectivePipeline:{round_id}] Processing batch {batch_num}/{total_batches} "
                f"({len(batch_urls)} URLs)"
            )

            # Scrape batch
            scraped_pages = await scrape_pages(self.scraper, batch_urls, round_id, self.log_manager)
            processed_urls.extend(batch_urls)

            if not scraped_pages:
                continue

            # Extract facts, entities, relations from batch
            batch_facts, batch_entities, batch_triples = await extract_all(
                self.fact_extractor,
                self.entity_extractor,
                self.relation_extractor,
                scraped_pages,
                round_id,
                self.log_manager,
            )

            if batch_facts:
                all_facts.extend(batch_facts)
                all_entities.extend(batch_entities)
                all_triples.extend(batch_triples)

                # Ingest to VDB and KG immediately
                await ingest_facts_and_triples(
                    self.vdb_ingest, self.kg_ingest, batch_facts, batch_triples, round_id, self.log_manager
                )

                # Re-retrieve and re-rank with new evidence
                dedup_sem, kg_candidates = await retrieve_candidates(
                    self.vdb_retriever,
                    self.kg_retriever,
                    queries,
                    list(set(all_entities)),
                    top_k * 2,
                    round_id,
                    self.log_manager,
                )

                top_ranked = await rank_candidates(
                    dedup_sem, kg_candidates, list(set(all_entities)), top_k, round_id, self.log_manager
                )

                top_score = top_ranked[0]["final_score"] if top_ranked else 0.0
                logger.info(
                    f"[CorrectivePipeline:{round_id}] After batch {batch_num}: "
                    f"top_score={top_score:.3f}, facts={len(all_facts)}"
                )

                # Check if threshold reached
                if top_score >= self.CONF_THRESHOLD:
                    logger.info(
                        f"[CorrectivePipeline:{round_id}] Trust threshold met after batch {batch_num}! "
                        f"({top_score:.3f} >= {self.CONF_THRESHOLD})"
                    )
                    break

        logger.info(
            f"[CorrectivePipeline:{round_id}] Completed: {len(all_facts)} facts, "
            f"{len(top_ranked)} ranked, processed {len(processed_urls)}/{len(search_urls)} URLs"
        )

        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Pipeline completed: {len(all_facts)} facts, {len(top_ranked)} ranked",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={
                    "facts_count": len(all_facts),
                    "triples_count": len(all_triples),
                    "ranked_count": len(top_ranked),
                    "top_score": top_ranked[0]["final_score"] if top_ranked else 0.0,
                    "urls_processed": len(processed_urls),
                    "urls_total": len(search_urls),
                },
            )

        return {
            "round_id": round_id,
            "status": "completed",
            "facts": all_facts,
            "triples": all_triples,
            "queries": queries,
            "semantic_candidates_count": len(dedup_sem),
            "kg_candidates_count": len(kg_candidates),
            "ranked": top_ranked,
            "used_web_search": True,
            "urls_processed": len(processed_urls),
        }

    async def _extract_claim_entities(self, claim: str, round_id: str) -> List[str]:
        """Extract entities from the claim text using LLM (1 call)."""
        try:
            # Use entity extractor on a synthetic fact
            synthetic_facts = [{"statement": claim, "source_url": "claim", "fact_id": "claim_0"}]
            annotated = await self.entity_extractor.annotate_entities(synthetic_facts)
            if annotated:
                return annotated[0].get("entities", [])
        except Exception as e:
            logger.warning(f"[CorrectivePipeline:{round_id}] Entity extraction from claim failed: {e}")
        return []


__all__ = ["CorrectivePipeline"]
