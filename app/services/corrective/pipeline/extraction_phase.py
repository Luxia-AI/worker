"""
Extraction Phase: Fact, entity, and relation extraction from scraped content.
"""

import os
import re
from hashlib import sha1
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger, is_debug_enabled, log_value_payload
from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.relation_extractor import RelationExtractor
from app.services.corrective.scraper import Scraper
from app.services.logging.log_manager import LogManager
from app.services.pipeline_debug_report import PipelineDebugReporter

logger = get_logger(__name__)


def _claim_context_hash(claim_text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(claim_text or "").strip().lower())
    if not normalized:
        return ""
    return sha1(normalized.encode("utf-8")).hexdigest()[:16]


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
    claim_text: str = "",
    claim_entities: Optional[List[str]] = None,
    must_have_entities: Optional[List[str]] = None,
    predicate_target: Optional[Dict[str, str]] = None,
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
    try:
        extracted_facts = await fact_extractor.extract(
            scraped_pages,
            predicate_target=predicate_target,
            claim_text=claim_text,
            claim_entities=claim_entities or [],
            must_have_entities=must_have_entities or [],
        )
    except TypeError as exc:
        # Backward compatibility for tests/mocks that still expose the legacy extractor signature.
        if "unexpected keyword argument" not in str(exc):
            raise
        try:
            extracted_facts = await fact_extractor.extract(
                scraped_pages,
                predicate_target=predicate_target,
            )
        except TypeError as exc2:
            if "unexpected keyword argument" not in str(exc2):
                raise
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

    # Stamp per-fact claim context so downstream ingest/retrieval can scope evidence to this claim.
    claim_hash = _claim_context_hash(claim_text)
    claim_ctx_entities: List[str] = []
    for ent in (must_have_entities or []) + (claim_entities or []):
        low = str(ent or "").strip().lower()
        if low and low not in claim_ctx_entities:
            claim_ctx_entities.append(low)
    claim_ctx_entities = claim_ctx_entities[:20]
    for fact in extracted_facts:
        if claim_hash and not str(fact.get("claim_context_hash") or "").strip():
            fact["claim_context_hash"] = claim_hash
        if claim_ctx_entities and not fact.get("claim_context_entities"):
            fact["claim_context_entities"] = list(claim_ctx_entities)
        if claim_ctx_entities and not fact.get("claim_entities_ctx"):
            fact["claim_entities_ctx"] = list(claim_ctx_entities)

    # 4) Entity annotation
    extracted_facts = await entity_extractor.annotate_entities(extracted_facts)
    all_entities = list({e for f in extracted_facts for e in (f.get("entities") or [])})
    # 5) Relation extraction
    try:
        triples = await relation_extractor.extract_relations(
            extracted_facts,
            all_entities,
            claim_text=claim_text,
            claim_entities=claim_entities or [],
            must_have_entities=must_have_entities or [],
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
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
