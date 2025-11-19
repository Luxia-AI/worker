import uuid
from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractingLLM
from app.services.corrective.relation_extractor import RelationExtractor
from app.services.corrective.scraper import Scraper
from app.services.corrective.trusted_search import TrustedSearch
from app.services.vdb.vdb_ingest import VDBIngest

logger = get_logger(__name__)


class CorrectivePipeline:
    """
    Corrective Retrieval Pipeline

    Executes a multi-agent evidence acquisition pipeline:
        1. Query Reformulation (inside TrustedSearch for now)
        2. Trusted Web Search
        3. Scraping & Content Extraction
        4. LLM Fact Extraction
        5. NER-based Entity Extraction
        6. LLM-based Relation Extraction
        7. Vector DB Ingestion (facts)
        8. KG Ingestion (triples)

    Returns enriched evidence blocks to the RAG worker orchestrator.
    """

    def __init__(self) -> None:
        self.search_agent = TrustedSearch()
        self.scraper = Scraper()
        self.fact_extractor = FactExtractingLLM()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.vdb_ingest = VDBIngest()

    async def run(
        self, post_text: str, domain: str, failed_entities: List[str] | None = None, round_id: str | None = None
    ) -> Dict[str, Any]:
        """
        Entry point for corrective retrieval.

        Args:
            post_text: The original user post.
            domain: Domain label (e.g., "health").
            failed_entities: Entities that failed initial KG/Vector DB checks.
            round_id: Unique ID for logging/debugging.

        Returns:
            Structured corrective evidence response.
        """

        round_id = round_id or str(uuid.uuid4())
        failed_entities = failed_entities or []

        logger.info(f"[CorrectivePipeline:{round_id}] Starting corrective retrieval")
        logger.info(f"[CorrectivePipeline:{round_id}] Post text: {post_text}")
        logger.info(f"[CorrectivePipeline:{round_id}] Failed entities: {failed_entities}")

        # ----------------------------------------
        # 1. Trusted Search
        # ----------------------------------------
        logger.info(f"[CorrectivePipeline:{round_id}] Running trusted search...")
        search_results = await self.search_agent.run(post_text, failed_entities)
        logger.info(f"[CorrectivePipeline:{round_id}] Search returned {len(search_results)} trusted URLs")

        if not search_results:
            return {
                "round_id": round_id,
                "facts": [],
                "triples": [],
                "errors": ["No trusted search results found"],
            }

        # ----------------------------------------
        # 2. Scrape URLs
        # ----------------------------------------
        logger.info(f"[CorrectivePipeline:{round_id}] Scraping {len(search_results)} URLs")
        scraped_pages = await self.scraper.scrape_all(search_results)
        logger.info(f"[CorrectivePipeline:{round_id}] Scraper extracted {len(scraped_pages)} pages")

        # Filter empty pages
        scraped_pages = [p for p in scraped_pages if p.get("content")]
        if not scraped_pages:
            return {
                "round_id": round_id,
                "facts": [],
                "triples": [],
                "errors": ["Scraper returned zero usable pages"],
            }

        # ----------------------------------------
        # 3. Extract Facts via LLM
        # ----------------------------------------
        logger.info(f"[CorrectivePipeline:{round_id}] Extracting atomic facts from pages...")
        # TODO: Implement fact extraction from pages using fact_extractor
        extracted_facts: List[Dict[str, Any]] = []
        for page in scraped_pages:
            # Extract facts from each page's content
            # This would involve calling the fact_extractor with the page content
            pass
        logger.info(f"[CorrectivePipeline:{round_id}] Extracted {len(extracted_facts)} facts")

        if not extracted_facts:
            return {
                "round_id": round_id,
                "facts": [],
                "triples": [],
                "errors": ["No facts extracted from pages"],
            }

        # ----------------------------------------
        # 4. Entity Extraction (SciSpaCy)
        # ----------------------------------------
        logger.info(f"[CorrectivePipeline:{round_id}] Running entity extraction...")
        extracted_facts = await self.entity_extractor.annotate_entities(extracted_facts)
        logger.info(f"[CorrectivePipeline:{round_id}] Entities annotated in {len(extracted_facts)} facts")

        # Gather unique entities for RE
        all_entities = list({ent for fact in extracted_facts for ent in fact.get("entities", [])})

        # ----------------------------------------
        # 5. Relation Extraction (LLM)
        # ----------------------------------------
        logger.info(f"[CorrectivePipeline:{round_id}] Extracting relations...")
        triples = await self.relation_extractor.extract_relations(facts=extracted_facts, entities=all_entities)
        logger.info(f"[CorrectivePipeline:{round_id}] Extracted {len(triples)} KG triples")

        # ----------------------------------------
        # 6. Ingest Into Vector DB
        # ----------------------------------------
        logger.info(f"[CorrectivePipeline:{round_id}] Ingesting facts into Vector DB...")
        await self.vdb_ingest.embed_and_ingest(extracted_facts)
        logger.info(f"[CorrectivePipeline:{round_id}] Vector DB ingestion complete")

        # ----------------------------------------
        # 7. Construct Response
        # ----------------------------------------
        logger.info(f"[CorrectivePipeline:{round_id}] Corrective retrieval complete")

        return {
            "round_id": round_id,
            "facts": extracted_facts,
            "triples": triples,
            "source_urls": search_results,
            "status": "completed",
        }
