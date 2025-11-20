# app/services/corrective/pipeline.py
import uuid
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.relation_extractor import RelationExtractor
from app.services.corrective.scraper import Scraper
from app.services.corrective.trusted_search import TrustedSearch
from app.services.kg.kg_ingest import KGIngest
from app.services.kg.kg_retrieval import KGRetrieval
from app.services.kg.neo4j_client import Neo4jClient
from app.services.ranking.hybrid_ranker import hybrid_rank
from app.services.vdb.vdb_ingest import VDBIngest
from app.services.vdb.vdb_retrieval import VDBRetrieval

logger = get_logger(__name__)


class CorrectivePipeline:
    """
    Corrective Retrieval Pipeline with integrated hybrid ranking.

    Flow:
        1. Query reformulation (TrustedSearch -> LLM)
        2. Trusted Search -> trusted URLs
        3. Scrape pages -> extract content
        4. Fact extraction (LLM)
        5. Entity extraction (LLM)
        6. Relation extraction (LLM)
        7. Ingest facts -> VDB (Pinecone) and KG (Neo4j)
        8. Retrieve candidates:
            - semantic results from VDB using each query
            - KG structural results using extracted entities
        9. Hybrid rank candidates and return top-K evidence
    """

    MAX_ROUNDS = 3
    CONF_THRESHOLD = 0.70  # if top evidence < 0.70; reinforce
    MIN_NEW_URLS = 2

    def __init__(self) -> None:
        self.search_agent = TrustedSearch()
        self.scraper = Scraper()
        self.fact_extractor = FactExtractor()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.vdb_ingest = VDBIngest()  # ingestion for new facts
        self.vdb_retriever = VDBRetrieval()  # retrieval for semantic candidates
        self.kg_client = Neo4jClient()  # used for lightweight KG retrieval
        self.kg_ingest = KGIngest()  # available if you want to ingest triples here
        self.kg_retriever = KGRetrieval()

    # ----------------------
    # Orchestrator run
    # ----------------------
    async def run(
        self,
        post_text: str,
        domain: str,
        failed_entities: Optional[List[str]] = None,
        round_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        round_id = round_id or str(uuid.uuid4())
        failed_entities = failed_entities or []

        logger.info(f"[CorrectivePipeline:{round_id}] Start for post: {post_text}")

        # 1) Trusted search: produce trusted URLs
        search_urls = await self.search_agent.run(post_text, failed_entities)
        logger.info(f"[CorrectivePipeline:{round_id}] Trusted search found {len(search_urls)} URLs")

        # 2) Scrape pages
        if search_urls:
            scraped_pages = await self.scraper.scrape_all(search_urls)
        else:
            scraped_pages = []
        scraped_pages = [p for p in scraped_pages if p.get("content")]
        logger.info(f"[CorrectivePipeline:{round_id}] Scraped {len(scraped_pages)} pages")

        # 3) Fact extraction
        extracted_facts = await self.fact_extractor.extract(scraped_pages)
        logger.info(f"[CorrectivePipeline:{round_id}] Extracted {len(extracted_facts)} facts")

        if not extracted_facts:
            return {"round_id": round_id, "facts": [], "triples": [], "ranked": [], "status": "no_facts"}

        # 4) Entity annotation (LLM-based/Scispacy fallback)
        extracted_facts = await self.entity_extractor.annotate_entities(extracted_facts)
        all_entities = list({e for f in extracted_facts for e in (f.get("entities") or [])})
        logger.info(f"[CorrectivePipeline:{round_id}] Extracted entities: {all_entities}")

        # 5) Relation extraction
        triples = await self.relation_extractor.extract_relations(extracted_facts, all_entities)
        logger.info(f"[CorrectivePipeline:{round_id}] Extracted {len(triples)} triples")

        # 6) Ingest facts/triples into stores (non-blocking best-effort)
        try:
            await self.vdb_ingest.embed_and_ingest(extracted_facts)
        except Exception as e:
            logger.warning(f"[CorrectivePipeline:{round_id}] VDB ingest failed: {e}")

        try:
            # KG ingest might be expensive; call but don't fail pipeline if it errors
            await self.kg_ingest.ingest_triples(triples)
        except Exception as e:
            logger.warning(f"[CorrectivePipeline:{round_id}] KG ingest failed: {e}")

        # 7) Candidate retrieval:
        semantic_candidates = []
        # Use top reformulated queries to retrieve semantic candidates
        # TrustedSearch.reformulate_queries might be on TrustedSearch; call it if available
        try:
            queries = (
                await self.search_agent.reformulate_queries(post_text, failed_entities)
                if hasattr(self.search_agent, "reformulate_queries")
                else [post_text]
            )
        except Exception:
            queries = [post_text]

        # run semantic retrieval for each query (aggregate)
        for q in queries:
            try:
                sem_res = await self.vdb_retriever.search(q, top_k=top_k)
                # sem_res expected format: list of dicts with 'statement','score',
                # 'entities','source_url','published_at','credibility'
                semantic_candidates.extend(sem_res or [])
            except Exception as e:
                logger.warning(f"[CorrectivePipeline:{round_id}] VDB retrieval failed for query='{q}': {e}")

        # dedupe semantic candidates by statement+source
        seen = set()
        dedup_sem = []
        for s in semantic_candidates:
            key = (s.get("statement"), s.get("source_url"))
            if key not in seen:
                seen.add(key)
                dedup_sem.append(s)

        # 8) KG candidates from KG retrieval using entities
        try:
            kg_candidates = await self.kg_retriever.retrieve(all_entities, top_k=top_k)
        except Exception as e:
            logger.warning(f"[CorrectivePipeline:{round_id}] KG retrieval failed: {e}")
            kg_candidates = []

        # 9) Hybrid ranking
        ranked = hybrid_rank(dedup_sem, kg_candidates, query_entities=all_entities)
        top_ranked = ranked[:top_k]

        # 10) Reinforcement loop if needed
        for round_count in range(1, self.MAX_ROUNDS + 1):
            score_str = top_ranked[0]["final_score"] if top_ranked else "N/A"
            logger.info(f"[CorrectivePipeline:{round_id}] Ranking round {round_count}, top score={score_str}")

            # If score meets threshold → stop reinforcement
            if top_ranked and top_ranked[0]["final_score"] >= self.CONF_THRESHOLD:
                logger.info(f"[CorrectivePipeline:{round_id}] Confidence threshold met. Stopping reinforcement.")
                break

            # Determine which items are weak
            low_conf_items = [r for r in ranked if r["final_score"] < self.CONF_THRESHOLD]
            if not low_conf_items:
                break

            # Reinforcement search
            new_urls = await self.search_agent.reinforce_search(low_conf_items, failed_entities)
            new_urls = [u for u in new_urls if u not in search_urls]

            if len(new_urls) < self.MIN_NEW_URLS:
                logger.info(f"[CorrectivePipeline:{round_id}] Not enough new URLs for reinforcement. Stopping.")
                break

            logger.info(f"[CorrectivePipeline:{round_id}] Reinforcement fetched {len(new_urls)} new URLs")

            search_urls.extend(new_urls)

            # Scrape new pages
            new_pages = await self.scraper.scrape_all(new_urls)
            new_pages = [p for p in new_pages if p.get("content")]

            # Extract new facts
            new_facts = await self.fact_extractor.extract(new_pages)
            if new_facts:
                new_facts = await self.entity_extractor.annotate_entities(new_facts)
                extracted_facts.extend(new_facts)

                # Extract triples from new facts
                new_triples = await self.relation_extractor.extract_relations(new_facts, all_entities)
                triples.extend(new_triples)

                # Ingest new facts
                try:
                    await self.vdb_ingest.embed_and_ingest(new_facts)
                except Exception:
                    pass
                try:
                    await self.kg_ingest.ingest_triples(new_triples)
                except Exception:
                    pass

            # Retrieve semantic again using new queries
            semantic_candidates = []
            for q in queries:
                try:
                    sem_res = await self.vdb_retriever.search(q, top_k=top_k)
                    semantic_candidates.extend(sem_res or [])
                except Exception:
                    pass

            # Deduplicate
            seen = set()
            dedup_sem = []
            for s in semantic_candidates:
                key = (s.get("statement"), s.get("source_url"))
                if key not in seen:
                    seen.add(key)
                    dedup_sem.append(s)

            # KG again
            try:
                kg_candidates = await self.kg_retriever.retrieve(all_entities, top_k=top_k)
            except Exception:
                kg_candidates = []

            # Re-rank
            ranked = hybrid_rank(dedup_sem, kg_candidates, query_entities=all_entities)
            top_ranked = ranked[:top_k]

            self.CONF_THRESHOLD -= round_count * 0.05

        logger.info(f"[CorrectivePipeline:{round_id}] Ranking produced {len(top_ranked)} top candidates")

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
