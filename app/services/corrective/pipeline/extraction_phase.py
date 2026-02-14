"""
Extraction Phase: Fact, entity, and relation extraction from scraped content.
"""

import os
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger, is_debug_enabled, log_value_payload
from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.relation_extractor import RelationExtractor
from app.services.corrective.scraper import Scraper
from app.services.logging.log_manager import LogManager
from app.services.pipeline_debug_report import PipelineDebugReporter

logger = get_logger(__name__)


async def scrape_pages(
    scraper: Scraper,
    urls: List[str],
    round_id: str,
    log_manager: Optional[LogManager] = None,
    debug_reporter: Optional[PipelineDebugReporter] = None,
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
    if debug_reporter:
        scrape_report = []
        for p in scraped_pages:
            scrape_report.append(
                {
                    "url": p.get("url"),
                    "was_scraping_skipped": not bool(p.get("content")),
                    "content_preview": (p.get("content", "")[:500] if p.get("content") else ""),
                    "source": p.get("source"),
                }
            )
        await debug_reporter.log_step(
            step_name="Scraping results",
            description="Was scraping skipped for each URL and scraped content preview",
            input_data={"urls": urls},
            output_data=scrape_report,
        )
    # Filter out pages with no content
    valid_pages = [p for p in scraped_pages if p.get("content")]
    log_value_payload(
        logger,
        "scrape",
        {
            "round_id": round_id,
            "urls_requested": len(urls or []),
            "pages_scraped": len(valid_pages or []),
            "scraped_urls": [p.get("url") for p in valid_pages if p.get("url")],
        },
    )

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"[PhaseOutput] scrape pages_scraped={len(valid_pages)}/{len(urls)}",
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
    debug_reporter: Optional[PipelineDebugReporter] = None,
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
    if debug_reporter:
        await debug_reporter.log_step(
            step_name="LLM-identified facts",
            description="Facts extracted from scraped contents",
            input_data={"scraped_pages_count": len(scraped_pages)},
            output_data=extracted_facts,
        )
    fact_confidences = [float(f.get("confidence") or 0.0) for f in extracted_facts if f.get("confidence") is not None]
    conf_min = min(fact_confidences) if fact_confidences else 0.0
    conf_max = max(fact_confidences) if fact_confidences else 0.0
    conf_avg = (sum(fact_confidences) / len(fact_confidences)) if fact_confidences else 0.0

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"[PhaseOutput] extraction facts_count={len(extracted_facts)}",
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
    # 5) Relation extraction
    triples = await relation_extractor.extract_relations(extracted_facts, all_entities)
    if debug_reporter:
        await debug_reporter.log_step(
            step_name="Knowledge graph tuples identified",
            description="Triples extracted from facts",
            input_data={"facts_count": len(extracted_facts), "entities_count": len(all_entities)},
            output_data=triples,
        )
    log_value_payload(
        logger,
        "extraction",
        {
            "round_id": round_id,
            "facts_sample": [
                {
                    "statement": f.get("statement", ""),
                    "source_url": f.get("source_url", ""),
                    "confidence": f.get("confidence"),
                }
                for f in extracted_facts
            ],
            "entities_sample": all_entities,
            "triples_sample": triples,
            "fact_conf_min": round(conf_min, 4),
            "fact_conf_max": round(conf_max, 4),
            "fact_conf_avg": round(conf_avg, 4),
        },
    )
    log_value_payload(
        logger,
        "extraction",
        {
            "round_id": round_id,
            "facts_all": extracted_facts,
            "entities_all": all_entities,
            "triples_all": triples,
        },
        level="debug",
        debug_only=True,
        sample_limit=int(os.getenv("LOG_VALUE_MAX_ITEMS", "20") or 20),
    )

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"[PhaseOutput] extraction triples_count={len(triples)} entities_count={len(all_entities)}",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={
                "phase": "extraction",
                "facts_sample": [f.get("statement", "") for f in extracted_facts[:5]],
                "entities_sample": all_entities[:10],
                "triples_sample": triples[:5],
                "fact_conf_stats": {"min": conf_min, "max": conf_max, "avg": conf_avg},
                "debug_mode": is_debug_enabled(),
            },
        )

    return extracted_facts, all_entities, triples
