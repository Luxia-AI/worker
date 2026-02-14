"""
Search Phase: Query reformulation and trusted domain search.
"""

from typing import List, Optional

from app.core.logger import get_logger, log_value_payload
from app.services.corrective.trusted_search import TrustedSearch
from app.services.logging.log_manager import LogManager

logger = get_logger(__name__)


async def do_search(
    search_agent: TrustedSearch,
    post_text: str,
    failed_entities: List[str],
    round_id: str,
    log_manager: Optional[LogManager] = None,
) -> tuple[List[str], List[str]]:
    """
    Execute search phase: reformulate queries and perform trusted search.

    Args:
        search_agent: TrustedSearch instance
        post_text: Original post text
        failed_entities: List of entities that failed extraction
        round_id: Round identifier for logging

    Returns:
        Tuple of (search_urls, reformulated_queries)
    """
    # 1) Reformulate queries using LLM
    try:
        queries = (
            await search_agent.reformulate_queries(post_text, failed_entities)
            if hasattr(search_agent, "reformulate_queries")
            else [post_text]
        )
    except Exception as e:
        logger.warning(f"[SearchPhase:{round_id}] Query reformulation failed: {e}")
        if log_manager:
            await log_manager.add_log(
                level="WARNING",
                message=f"Query reformulation failed: {e}",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
            )
        queries = [post_text]

    # 2) Trusted search: produce trusted URLs
    search_urls = await search_agent.run(post_text, failed_entities)
    providers_used = []
    if getattr(search_agent, "google_available", False):
        providers_used.append("google")
    if getattr(search_agent, "serper_available", False):
        providers_used.append("serper")
    if not providers_used:
        providers_used.append("unknown")
    log_value_payload(
        logger,
        "search",
        {
            "round_id": round_id,
            "queries_total": len(queries or []),
            "queries_executed": len(queries or []),
            "queries_used": queries or [],
            "providers_used": providers_used,
            "urls_found_total": len(search_urls or []),
            "trusted_urls": search_urls or [],
        },
    )

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"[PhaseOutput] search urls_found_total={len(search_urls)} queries_total={len(queries)}",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={
                "phase": "search",
                "queries_total": len(queries or []),
                "queries_used": queries or [],
                "urls_found_total": len(search_urls or []),
                "trusted_urls": search_urls[:5] if search_urls else [],
            },
        )

    return search_urls, queries
