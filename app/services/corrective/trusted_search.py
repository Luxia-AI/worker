import asyncio
from typing import List

import aiohttp

from app.core.config import settings
from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService

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

QUERY_REFORMULATION_PROMPT = """
You are a query reformulation agent specialized in retrieving
high-quality medical evidence from trusted sources (CDC, NIH, WHO, Mayo Clinic, Harvard Health).

Given a social media post and optional failed biomedical entities:

1. Extract the medically relevant keywords.
2. Generate 5-8 optimized Google search queries.
3. Each query must be:
   - short (3-7 words)
   - keyword dense
   - objective
   - medically oriented
   - suitable for evidence gathering
4. Avoid question-like queries. Focus on **search-efficient** queries.

Return ONLY this JSON structure:
{
  "queries": ["...", "...", "..."]
}
"""


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

    GOOGLE_API_KEY = settings.GOOGLE_API_KEY
    GOOGLE_CSE_ID = settings.GOOGLE_CSE_ID
    MAX_RESULTS_PER_QUERY = 6

    SEARCH_URL = "https://www.googleapis.com/customsearch/v1?" "key={key}&cx={cse}&q={query}"

    def __init__(self) -> None:
        if not self.GOOGLE_API_KEY or not self.GOOGLE_CSE_ID:
            logger.error("Google Search API or CSE ID missing from environment")
            raise RuntimeError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")

        self.groq_client = GroqService()

    # ---------------------------------------------------------------------
    # Query Reformulation
    # ---------------------------------------------------------------------
    async def reformulate_queries(self, text: str, failed_entities: List[str]) -> List[str]:
        """
        LLM-powered search query reformulation.
        Produces highly optimized evidence retrieval phrases.
        """
        prompt = f"""
{QUERY_REFORMULATION_PROMPT}

POST TEXT:
{text}

FAILED ENTITIES:
{failed_entities}
"""

        try:
            result = await self.groq_client.ainvoke(prompt, response_format="json")
            queries = result.get("queries", [])
            cleaned = [q.strip().lower() for q in queries if isinstance(q, str)]
            return list(dict.fromkeys(cleaned))  # dedupe but preserve order

        except Exception as e:
            logger.error(f"[TrustedSearch] LLM query reformulation failed: {e}")
            # fallback → old heuristic
            return self._fallback_queries(text, failed_entities)

    def _fallback_queries(self, text: str, failed_entities: List[str]) -> List[str]:
        base = text.lower()
        queries = [
            f"{base} medical evidence",
            f"{base} scientific research",
            f"{base} clinical verification",
            f"{base} factual analysis",
        ]

        for ent in failed_entities or []:
            queries.append(f"{ent} medical evidence")
            queries.append(f"{ent} health research")
            queries.append(f"{ent} scientific facts")

        return list(dict.fromkeys(queries))

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
        queries = await self.reformulate_queries(post_text, failed_entities)
        logger.info(f"[TrustedSearch] Reformulated queries: {queries}")

        async with aiohttp.ClientSession() as session:
            tasks = [self.search_query(session, q) for q in queries]
            results = await asyncio.gather(*tasks)

        # Flatten & dedupe
        urls = {url for sublist in results for url in sublist}

        logger.info(f"[TrustedSearch] Found {len(urls)} trusted URLs")
        return list(urls)
