"""
Reinforcement Phase: Low-confidence reinforcement search loop.
"""

from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.pipeline.extraction_phase import extract_all, scrape_pages
from app.services.corrective.pipeline.ingestion_phase import ingest_facts_and_triples
from app.services.corrective.pipeline.ranking_phase import rank_candidates
from app.services.corrective.pipeline.retrieval_phase import retrieve_candidates
from app.services.corrective.relation_extractor import RelationExtractor
from app.services.corrective.scraper import Scraper
from app.services.corrective.trusted_search import TrustedSearch
from app.services.kg.kg_ingest import KGIngest
from app.services.vdb.vdb_ingest import VDBIngest

logger = get_logger(__name__)


async def reinforcement_loop(
    search_agent: TrustedSearch,
    scraper: Scraper,
    fact_extractor: FactExtractor,
    entity_extractor: EntityExtractor,
    relation_extractor: RelationExtractor,
    vdb_ingest: VDBIngest,
    kg_ingest: KGIngest,
    vdb_retriever: Any,  # VDBRetrieval
    kg_retriever: Any,  # KGRetrieval
    search_urls: List[str],
    extracted_facts: List[Dict[str, Any]],
    triples: List[Dict[str, Any]],
    all_entities: List[str],
    queries: List[str],
    ranked: List[Dict[str, Any]],
    post_text: str,
    failed_entities: List[str],
    top_k: int,
    max_rounds: int,
    conf_threshold: float,
    min_new_urls: int,
    round_id: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Execute reinforcement loop for low-confidence results.

    Args:
        search_agent: TrustedSearch instance
        scraper: Scraper instance
        fact_extractor: FactExtractor instance
        entity_extractor: EntityExtractor instance
        relation_extractor: RelationExtractor instance
        vdb_ingest: VDBIngest instance
        kg_ingest: KGIngest instance
        vdb_retriever: VDBRetrieval instance
        kg_retriever: KGRetrieval instance
        search_urls: Current list of search URLs
        extracted_facts: Accumulated facts
        triples: Accumulated triples
        all_entities: All extracted entities
        queries: List of search queries
        ranked: Currently ranked candidates
        post_text: Original post text
        failed_entities: Entities that failed extraction
        top_k: Number of top candidates
        max_rounds: Maximum reinforcement rounds
        conf_threshold: Confidence threshold
        min_new_urls: Minimum new URLs per round
        round_id: Round identifier for logging

    Returns:
        Tuple of (extracted_facts, triples, top_ranked)
    """
    top_ranked = ranked[:top_k]

    for round_count in range(1, max_rounds + 1):
        score_str = top_ranked[0]["final_score"] if top_ranked else "N/A"
        logger.info(f"[ReinforcementPhase:{round_id}] Round {round_count}, top score={score_str}")

        # If score meets threshold → stop reinforcement
        if top_ranked and top_ranked[0]["final_score"] >= conf_threshold:
            logger.info(f"[ReinforcementPhase:{round_id}] Confidence threshold met. Stopping reinforcement.")
            break

        # Determine which items are weak
        low_conf_items = [r for r in ranked if r["final_score"] < conf_threshold]
        if not low_conf_items:
            break

        # Reinforcement search
        new_urls = await search_agent.reinforce_search(low_conf_items, failed_entities)
        new_urls = [u for u in new_urls if u not in search_urls]

        if len(new_urls) < min_new_urls:
            logger.info(
                f"[ReinforcementPhase:{round_id}] Not enough new URLs " f"({len(new_urls)} < {min_new_urls}). Stopping."
            )
            break

        logger.info(f"[ReinforcementPhase:{round_id}] Fetched {len(new_urls)} new URLs")
        search_urls.extend(new_urls)

        # Scrape and extract from new pages
        new_pages = await scrape_pages(scraper, new_urls, round_id)
        if new_pages:
            new_facts, _, new_triples = await extract_all(
                fact_extractor, entity_extractor, relation_extractor, new_pages, round_id
            )

            if new_facts:
                extracted_facts.extend(new_facts)
                triples.extend(new_triples)

                # Ingest new data
                await ingest_facts_and_triples(vdb_ingest, kg_ingest, new_facts, new_triples, round_id)

        # Retrieve and re-rank
        dedup_sem, kg_candidates = await retrieve_candidates(
            vdb_retriever, kg_retriever, queries, all_entities, top_k, round_id
        )
        top_ranked = await rank_candidates(dedup_sem, kg_candidates, all_entities, top_k, round_id)

        # Decay confidence threshold
        conf_threshold -= round_count * 0.05

    return extracted_facts, triples, top_ranked
