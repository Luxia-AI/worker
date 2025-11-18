import asyncio
from typing import List

import aiohttp

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Trusted medical domains
TRUSTED_DOMAINS = {
    "who.int",
    "cdc.gov",
    "nih.gov",
    "fda.gov",
    "nhs.uk",
    "mayoclinic.org",
    "health.harvard.edu",
    "medlineplus.gov",
    "livescience.com",
    "medicalnewstoday.com",
}


class TrustedSearch:
    """
    Google Custom Search (CSE) integration for trusted domain retrieval.

    Provides:
        - automated query reformulation
        - domain-filtered search results
        - async network requests
        - configurable CSE and API keys
        - structured return format
    """

    GOOGLE_API_KEY = settings.google_api_key
    GOOGLE_CSE_ID = settings.google_cse_id
    MAX_RESULTS_PER_QUERY = 6

    SEARCH_URL = "https://www.googleapis.com/customsearch/v1?" "key={key}&cx={cse}&q={query}"

    def __init__(self):
        if not self.GOOGLE_API_KEY or not self.GOOGLE_CSE_ID:
            logger.error("Google Search API or CSE ID missing from environment")
            raise RuntimeError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")

    # ---------------------------------------------------------------------
    # Query Reformulation (Light Agent)
    # ---------------------------------------------------------------------
    def reformulate_queries(self, text: str, failed_entities: List[str]) -> List[str]:
        """
        Lightweight heuristic query reformulation.
        LLM-based reformulation will come later, but this is good for stage 1.

        Generates 4–6 angle-shifted search queries.
        """
        base = text.lower()

        queries = [
            f"{base} medical evidence",
            f"{base} scientific research",
            f"{base} clinical verification",
            f"{base} is it true? health facts",
        ]

        if failed_entities:
            for ent in failed_entities:
                queries.append(f"{ent} evidence medical study")
                queries.append(f"{ent} health facts scientific")

        return list(set(queries))  # remove duplicates

    # ---------------------------------------------------------------------
    # Single Query Search
    # ---------------------------------------------------------------------
    async def search_query(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """
        Runs a single Google CSE request and returns trusted URLs.
        """

        url = self.SEARCH_URL.format(
            key=self.GOOGLE_API_KEY,
            cse=self.GOOGLE_CSE_ID,
            query=query.replace(" ", "+"),
        )

        try:
            async with session.get(url, timeout=12) as resp:
                data = await resp.json()

                if "items" not in data:
                    return []

                urls = []
                for item in data["items"]:
                    link = item.get("link")
                    if link and self.is_trusted(link):
                        urls.append(link)

                return urls

        except Exception as e:
            logger.error(f"[TrustedSearch] Google search failed for query='{query}': {e}")
            return []

    # ---------------------------------------------------------------------
    # Domain Whitelisting
    # ---------------------------------------------------------------------
    def is_trusted(self, url: str) -> bool:
        """
        Returns True only if domain is in trusted domain list.
        """
        try:
            domain = url.split("/")[2]
            return domain in TRUSTED_DOMAINS
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Main Search Handler
    # ---------------------------------------------------------------------
    async def run(self, post_text: str, failed_entities: List[str]) -> List[str]:
        """
        Executes reformulation → parallelized Google CSE queries → URL dedupe.
        """
        queries = self.reformulate_queries(post_text, failed_entities)
        logger.info(f"[TrustedSearch] Reformulated queries: {queries}")

        async with aiohttp.ClientSession() as session:
            tasks = [self.search_query(session, q) for q in queries]
            results = await asyncio.gather(*tasks)

        # Flatten & dedupe
        urls = {url for sublist in results for url in sublist}

        logger.info(f"[TrustedSearch] Found {len(urls)} trusted URLs")
        return list(urls)
