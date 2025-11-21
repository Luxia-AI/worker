"""
Extraction Phase: Fact, entity, and relation extraction from scraped content.
"""

from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.relation_extractor import RelationExtractor
from app.services.corrective.scraper import Scraper
from app.services.logging.log_manager import LogManager

logger = get_logger(__name__)


async def scrape_pages(
    scraper: Scraper,
    urls: List[str],
    round_id: str,
    log_manager: Optional[LogManager] = None,
) -> List[Dict[str, Any]]:
    """
    Scrape pages from list of URLs.

    Args:
        scraper: Scraper instance
        urls: List of URLs to scrape
        round_id: Round identifier for logging

    Returns:
        List of scraped pages with content
    """
    if not urls:
        return []

    scraped_pages = await scraper.scrape_all(urls)
    # Filter out pages with no content
    valid_pages = [p for p in scraped_pages if p.get("content")]
    logger.info(f"[ExtractionPhase:{round_id}] Scraped {len(valid_pages)} pages from {len(urls)} URLs")

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"Scraping completed: {len(valid_pages)}/{len(urls)} pages extracted",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={"valid_pages": len(valid_pages), "total_urls": len(urls)},
        )

    return valid_pages


async def extract_all(
    fact_extractor: FactExtractor,
    entity_extractor: EntityExtractor,
    relation_extractor: RelationExtractor,
    scraped_pages: List[Dict[str, Any]],
    round_id: str,
    log_manager: Optional[LogManager] = None,
) -> tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    """
    Extract facts, entities, and relations from scraped pages.

    Args:
        fact_extractor: FactExtractor instance
        entity_extractor: EntityExtractor instance
        relation_extractor: RelationExtractor instance
        scraped_pages: List of scraped page dicts
        round_id: Round identifier for logging

    Returns:
        Tuple of (extracted_facts, all_entities, extracted_triples)
    """
    # 3) Fact extraction
    extracted_facts = await fact_extractor.extract(scraped_pages)
    logger.info(f"[ExtractionPhase:{round_id}] Extracted {len(extracted_facts)} facts")

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"Fact extraction completed: {len(extracted_facts)} facts",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={"facts_count": len(extracted_facts)},
        )

    if not extracted_facts:
        return [], [], []

    # 4) Entity annotation
    extracted_facts = await entity_extractor.annotate_entities(extracted_facts)
    all_entities = list({e for f in extracted_facts for e in (f.get("entities") or [])})
    logger.info(f"[ExtractionPhase:{round_id}] Extracted {len(all_entities)} entities")

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"Entity extraction completed: {len(all_entities)} entities",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={"entities_count": len(all_entities)},
        )

    # 5) Relation extraction
    triples = await relation_extractor.extract_relations(extracted_facts, all_entities)
    logger.info(f"[ExtractionPhase:{round_id}] Extracted {len(triples)} triples")

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"Relation extraction completed: {len(triples)} triples",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={"triples_count": len(triples)},
        )

    return extracted_facts, all_entities, triples
