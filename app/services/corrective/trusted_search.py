import asyncio
from typing import Any, Dict, List

import aiohttp

from app.constants.config import GOOGLE_CSE_SEARCH_URL, GOOGLE_CSE_TIMEOUT, TRUSTED_DOMAINS
from app.constants.llm_prompts import QUERY_REFORMULATION_PROMPT, REINFORCEMENT_QUERY_PROMPT
from app.core.config import settings
from app.core.logger import get_logger
from app.core.rate_limit import throttled
from app.services.common.list_ops import dedupe_list
from app.services.common.url_helpers import dedup_urls, is_accessible_url
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)


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

    SEARCH_URL = GOOGLE_CSE_SEARCH_URL

    def __init__(self) -> None:
        if not self.GOOGLE_API_KEY or not self.GOOGLE_CSE_ID:
            logger.error("Google Search API or CSE ID missing from environment")
            raise RuntimeError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")

        self.llm_client = HybridLLMService()

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
            # HIGH priority: Query reformulation is crucial for good search results
            result = await self.llm_client.ainvoke(prompt, response_format="json", priority=LLMPriority.HIGH)
            queries = result.get("queries", [])
            cleaned = [q.strip().lower() for q in queries if isinstance(q, str)]
            return dedupe_list(cleaned)  # dedupe but preserve order

        except Exception as e:
            logger.error(f"[TrustedSearch] LLM query reformulation failed: {e}")
            # fallback → old heuristic
            return self._fallback_queries(text, failed_entities)

    # ---------------------------------------------------------------------
    # Fallback Query Generation
    # ---------------------------------------------------------------------
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

        return dedupe_list(queries)

    # ---------------------------------------------------------------------
    # Single Query Search
    # ---------------------------------------------------------------------
    @throttled(limit=100, period=60.0, name="google_cse")
    async def search_query(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """
        Runs a single Google CSE request and returns trusted URLs.
        """
        import urllib.parse

        # Properly encode the query
        encoded_query = urllib.parse.quote_plus(query)

        url = self.SEARCH_URL.format(
            key=self.GOOGLE_API_KEY,
            cse=self.GOOGLE_CSE_ID,
            query=encoded_query,
        )

        try:
            async with session.get(url, timeout=GOOGLE_CSE_TIMEOUT) as resp:
                data = await resp.json()

                # Log full response for debugging
                if "error" in data:
                    logger.error(
                        f"[TrustedSearch] Google API error for '{query}': "
                        f"{data['error'].get('message', data['error'])}"
                    )
                    return []

                if "items" not in data:
                    # Log search info if available
                    search_info = data.get("searchInformation", {})
                    total_results = search_info.get("totalResults", "0")
                    logger.warning(f"[TrustedSearch] No items for query='{query}' " f"(totalResults={total_results})")
                    return []

                # Log all returned URLs for debugging
                all_urls = [item.get("link", "N/A") for item in data["items"]]
                logger.info(f"[TrustedSearch] Query '{query}' raw results: {all_urls[:5]}...")

                urls = []
                for item in data["items"]:
                    link = item.get("link")
                    if link:
                        if self.is_trusted(link):
                            urls.append(link)
                            logger.info(f"[TrustedSearch] ✓ Accepted: {link}")
                        else:
                            logger.debug(f"[TrustedSearch] ✗ Filtered: {link}")

                logger.info(f"[TrustedSearch] Query '{query}' -> " f"{len(urls)}/{len(data['items'])} trusted URLs")
                return urls

        except asyncio.TimeoutError:
            logger.error(f"[TrustedSearch] Timeout for query='{query}'")
            return []
        except Exception as e:
            logger.error(f"[TrustedSearch] Search failed for query='{query}': {e}")
            return []

    # ---------------------------------------------------------------------
    # Domain Whitelisting
    # ---------------------------------------------------------------------
    def is_trusted(self, url: str) -> bool:
        """
        Returns True if domain is in trusted domain list or is a .gov/.edu domain.
        """
        if not is_accessible_url(url):
            return False
        try:
            domain = url.split("/")[2].lower()

            # Check exact domain match
            if domain in TRUSTED_DOMAINS:
                return True

            # Check if any trusted domain is a suffix (handles subdomains)
            for trusted in TRUSTED_DOMAINS:
                if domain.endswith(trusted) or domain.endswith("." + trusted):
                    return True

            # Accept any .gov or .edu domain (very trustworthy)
            if domain.endswith(".gov") or domain.endswith(".edu"):
                return True

            return False
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Site-Specific Query Generation
    # ---------------------------------------------------------------------
    def _generate_site_queries(self, base_query: str) -> List[str]:
        """
        Generate queries with site: operator for trusted domains.
        This ensures Google returns results from authoritative sources.
        """
        priority_sites = [
            "nih.gov",
            "cdc.gov",
            "who.int",
            "mayoclinic.org",
            "pubmed.ncbi.nlm.nih.gov",
            "health.harvard.edu",
        ]

        site_queries = []
        for site in priority_sites:
            site_queries.append(f"site:{site} {base_query}")

        return site_queries

    # ---------------------------------------------------------------------
    # Main Search Handler
    # ---------------------------------------------------------------------
    async def run(
        self,
        post_text: str,
        failed_entities: List[str],
        min_urls: int = 3,
        max_queries: int = 15,
    ) -> List[str]:
        """
        Executes reformulation → sequential Google CSE queries → URL dedupe.
        Stops early once min_urls threshold is reached.
        """
        # 1) Generate reformulated queries
        queries = await self.reformulate_queries(post_text, failed_entities)
        logger.info(f"[TrustedSearch] Reformulated queries: {queries}")

        # 2) Also generate site-specific queries for better targeting
        if queries:
            # Use first query as base for site-specific searches
            site_queries = self._generate_site_queries(queries[0])
            # Interleave: original, site-specific, original, site-specific...
            all_queries = []
            for i, q in enumerate(queries):
                all_queries.append(q)
                if i < len(site_queries):
                    all_queries.append(site_queries[i])
            # Add remaining site queries
            all_queries.extend(site_queries[len(queries) :])
        else:
            all_queries = queries

        # Limit total queries
        all_queries = all_queries[:max_queries]
        logger.info(f"[TrustedSearch] Total queries to try: {len(all_queries)}")

        # 3) Execute queries sequentially until we have enough URLs
        collected_urls: set[str] = set()

        async with aiohttp.ClientSession() as session:
            for i, query in enumerate(all_queries):
                logger.info(f"[TrustedSearch] Executing query {i + 1}/{len(all_queries)}: '{query}'")

                try:
                    urls = await self.search_query(session, query)
                    collected_urls.update(urls)

                    logger.info(f"[TrustedSearch] Progress: {len(collected_urls)} URLs " f"(need {min_urls})")

                    # Early exit if we have enough
                    if len(collected_urls) >= min_urls:
                        logger.info(
                            f"[TrustedSearch] Threshold reached ({len(collected_urls)} >= "
                            f"{min_urls}), stopping search"
                        )
                        break

                except Exception as e:
                    logger.warning(f"[TrustedSearch] Query '{query}' failed: {e}")
                    continue

        # 4) Dedupe and return
        urls = dedup_urls(list(collected_urls))
        logger.info(f"[TrustedSearch] Final result: {len(urls)} trusted URLs")
        return list(urls)

    async def google_search(self, post_text: str, failed_entities: List[str] | None = None) -> List[str]:
        """
        Alias for run() - performs Google CSE search with query reformulation.
        """
        if failed_entities is None:
            failed_entities = []
        return await self.run(post_text, failed_entities)

    # ---------------------------------------------------------------------
    # LLM-powered Reinforced Query Generation
    # ---------------------------------------------------------------------
    async def llm_reformulate_for_reinforcement(
        self,
        low_conf_items: List[Dict[str, Any]],
        failed_entities: List[str],
    ) -> List[str]:
        """
        Uses LLM (Groq with Ollama fallback) to generate highly optimized reinforcement search queries.
        Much better than heuristic string concatenation.
        """

        base_statements = [item.get("statement", "") for item in low_conf_items]
        base_entities = failed_entities or []

        prompt = REINFORCEMENT_QUERY_PROMPT.format(statements=base_statements, entities=base_entities)

        try:
            # HIGH priority: Reinforcement query generation is crucial for finding evidence
            result = await self.llm_client.ainvoke(prompt, response_format="json", priority=LLMPriority.HIGH)
            queries = result.get("queries", [])

            # safety: ensure list[str]
            return [q for q in queries if isinstance(q, str) and q.strip()]

        except Exception as e:
            logger.warning(f"[TrustedSearch] LLM reinforcement query generation failed: {e}")

            # fallback to simple heuristics
            fallback = []
            for stmt in base_statements:
                fallback.append(f"{stmt} peer reviewed research")
                fallback.append(f"{stmt} scientific study NIH CDC")

            for ent in base_entities:
                fallback.append(f"{ent} medical research")
                fallback.append(f"{ent} clinical facts verified")

            return dedupe_list(fallback)

    # ---------------------------------------------------------------------
    # Reinforced Search for Low Confidence Cases
    # ---------------------------------------------------------------------
    async def reinforce_search(
        self,
        low_conf_items: List[Dict[str, Any]],
        failed_entities: List[str],
        max_queries: int = 8,
    ) -> List[str]:
        """
        LLM-powered reinforcement search using Kimi/Groq.
        Calls Google CSE on the generated queries.
        """

        # 1) Generate smarter queries
        queries = await self.llm_reformulate_for_reinforcement(low_conf_items, failed_entities)
        queries = queries[:max_queries]  # cap

        logger.info(f"[TrustedSearch] Reinforcement queries: {queries}")

        # 2) Perform Google CSE search
        all_urls = []
        async with aiohttp.ClientSession() as session:
            for q in queries:
                try:
                    urls = await self.search_query(session, q)
                    all_urls.extend(urls)
                except Exception as e:
                    logger.warning(f"[TrustedSearch] Reinforcement search failed for '{q}': {e}")

        # 3) Deduplicate
        return dedup_urls(all_urls)
