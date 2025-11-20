"""
Search Phase: Query reformulation and trusted domain search.
"""

from typing import List

from app.core.logger import get_logger
from app.services.corrective.trusted_search import TrustedSearch

logger = get_logger(__name__)


async def do_search(
    search_agent: TrustedSearch,
    post_text: str,
    failed_entities: List[str],
    round_id: str,
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
        queries = [post_text]

    # 2) Trusted search: produce trusted URLs
    search_urls = await search_agent.run(post_text, failed_entities)
    logger.info(f"[SearchPhase:{round_id}] Trusted search found {len(search_urls)} URLs")

    return search_urls, queries
