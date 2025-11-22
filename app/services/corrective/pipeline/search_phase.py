"""
Search Phase: Query reformulation and trusted domain search.
"""

from typing import List, Optional

from app.core.logger import get_logger
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
    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message="Starting search phase: query reformulation",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={"post_text": post_text[:100], "failed_entities": failed_entities},
        )

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
    logger.info(f"[SearchPhase:{round_id}] Trusted search found {len(search_urls)} URLs")

    if log_manager:
        await log_manager.add_log(
            level="INFO",
            message=f"Search phase completed: {len(search_urls)} URLs found",
            module=__name__,
            request_id=f"claim-{round_id}",
            round_id=round_id,
            context={"url_count": len(search_urls), "queries": queries},
        )

    return search_urls, queries
