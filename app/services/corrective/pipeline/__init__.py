"""
Corrective Retrieval Pipeline: Main orchestrator.

Coordinates multiple phases:
    1. Search: Query reformulation and trusted domain search
    2. Extraction: Fact, entity, and relation extraction
    3. Ingestion: VDB and KG persistence
    4. Retrieval: Semantic and KG candidate retrieval
    5. Ranking: Hybrid ranking of candidates
    6. Reinforcement: Low-confidence reinforcement loop
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
from app.services.corrective.pipeline.reinforcement_phase import reinforcement_loop
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
        Execute the full corrective retrieval pipeline.

        Args:
            post_text: Original post/claim text
            domain: Domain/source of post
            failed_entities: Entities that failed extraction
            round_id: Unique round identifier (auto-generated if None)
            top_k: Number of top candidates to return

        Returns:
            Pipeline result dict with facts, triples, queries, and ranked evidence
        """
        round_id = round_id or str(uuid.uuid4())
        failed_entities = failed_entities or []

        # Reset Groq call counter for this new request (max 3 Groq calls, then Ollama)
        reset_groq_counter()

        logger.info(f"[CorrectivePipeline:{round_id}] Start for post: {post_text}")

        # Log to LogManager if available
        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Pipeline started for claim: {post_text[:100]}...",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={"domain": domain, "failed_entities": failed_entities},
            )

        # Phase 1: Search
        search_urls, queries = await do_search(
            self.search_agent, post_text, failed_entities, round_id, self.log_manager
        )

        # Phase 2: Scrape pages
        scraped_pages = await scrape_pages(self.scraper, search_urls, round_id, self.log_manager)

        # Phase 3: Extract facts, entities, relations
        extracted_facts, all_entities, triples = await extract_all(
            self.fact_extractor,
            self.entity_extractor,
            self.relation_extractor,
            scraped_pages,
            round_id,
            self.log_manager,
        )

        if not extracted_facts:
            return {"round_id": round_id, "facts": [], "triples": [], "ranked": [], "status": "no_facts"}

        # Phase 4: Ingest to VDB and KG
        await ingest_facts_and_triples(
            self.vdb_ingest, self.kg_ingest, extracted_facts, triples, round_id, self.log_manager
        )

        # Phase 5: Retrieve semantic and KG candidates
        dedup_sem, kg_candidates = await retrieve_candidates(
            self.vdb_retriever,
            self.kg_retriever,
            queries,
            all_entities,
            top_k,
            round_id,
            self.log_manager,
        )

        # Phase 6: Hybrid ranking
        top_ranked = await rank_candidates(dedup_sem, kg_candidates, all_entities, top_k, round_id, self.log_manager)

        # Phase 7: Reinforcement loop (if needed)
        extracted_facts, triples, top_ranked = await reinforcement_loop(
            self.search_agent,
            self.scraper,
            self.fact_extractor,
            self.entity_extractor,
            self.relation_extractor,
            self.vdb_ingest,
            self.kg_ingest,
            self.vdb_retriever,
            self.kg_retriever,
            search_urls,
            extracted_facts,
            triples,
            all_entities,
            queries,
            top_ranked,
            post_text,
            failed_entities,
            top_k,
            self.MAX_ROUNDS,
            self.CONF_THRESHOLD,
            self.MIN_NEW_URLS,
            round_id,
            self.log_manager,
        )

        logger.info(f"[CorrectivePipeline:{round_id}] Completed with {len(top_ranked)} top-ranked evidence")

        # Log completion to LogManager
        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Pipeline completed: {len(extracted_facts)} facts, {len(top_ranked)} ranked results",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={
                    "facts_count": len(extracted_facts),
                    "triples_count": len(triples),
                    "ranked_count": len(top_ranked),
                    "top_score": top_ranked[0]["final_score"] if top_ranked else 0.0,
                },
            )

        return {
            "round_id": round_id,
            "status": "completed",
            "facts": extracted_facts,
            "triples": triples,
            "queries": queries,
            "semantic_candidates_count": len(dedup_sem),
            "kg_candidates_count": len(kg_candidates),
            "ranked": top_ranked,
        }


__all__ = ["CorrectivePipeline"]
