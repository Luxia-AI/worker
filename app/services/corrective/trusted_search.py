import asyncio
from typing import Any, Dict, List, Tuple

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


class QuotaExceededError(Exception):
    """Raised when search API quota is exceeded."""

    pass


class TrustedSearch:
    """
    Google Custom Search (CSE) integration for trusted domain retrieval.
    Falls back to Serper.dev if Google quota is exceeded.

    Provides:
        - automated query reformulation
        - domain-filtered search results
        - async network requests
        - configurable CSE and API keys
        - Serper.dev fallback for quota issues
        - structured return format
    """

    GOOGLE_API_KEY = settings.GOOGLE_API_KEY
    GOOGLE_CSE_ID = settings.GOOGLE_CSE_ID
    SERPER_API_KEY = settings.SERPER_API_KEY

    SEARCH_URL = GOOGLE_CSE_SEARCH_URL
    SERPER_URL = "https://google.serper.dev/search"

    def __init__(self) -> None:
        # Allow initialization even without Google API if Serper is available
        has_google = self.GOOGLE_API_KEY and self.GOOGLE_CSE_ID
        has_serper = bool(self.SERPER_API_KEY)

        if not has_google and not has_serper:
            logger.error("No search API configured (need GOOGLE_API_KEY+CSE_ID or SERPER_API_KEY)")
            raise RuntimeError("Missing search API credentials")

        self.google_available = has_google
        self.serper_available = has_serper
        self.google_quota_exceeded = False

        if has_google:
            logger.info("[TrustedSearch] Google CSE configured")
        if has_serper:
            logger.info("[TrustedSearch] Serper.dev fallback configured")

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
    # Single Query Search (Google CSE)
    # ---------------------------------------------------------------------
    @throttled(limit=100, period=60.0, name="google_cse")
    async def search_query_google(self, session: aiohttp.ClientSession, query: str) -> Tuple[List[str], bool]:
        """
        Runs a single Google CSE request and returns trusted URLs.
        Returns (urls, quota_exceeded) tuple.
        """
        import urllib.parse

        encoded_query = urllib.parse.quote_plus(query)

        url = self.SEARCH_URL.format(
            key=self.GOOGLE_API_KEY,
            cse=self.GOOGLE_CSE_ID,
            query=encoded_query,
        )

        try:
            async with session.get(url, timeout=GOOGLE_CSE_TIMEOUT) as resp:
                data = await resp.json()

                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    logger.error(f"[TrustedSearch:Google] API error: {error_msg}")

                    # Detect quota exceeded
                    if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                        return [], True  # Signal quota exceeded

                    return [], False

                if "items" not in data:
                    search_info = data.get("searchInformation", {})
                    total = search_info.get("totalResults", "0")
                    logger.warning(f"[TrustedSearch:Google] No items for '{query}' (total={total})")
                    return [], False

                all_urls = [item.get("link", "N/A") for item in data["items"]]
                logger.info(f"[TrustedSearch:Google] '{query}' raw: {all_urls[:3]}...")

                urls = []
                for item in data["items"]:
                    link = item.get("link")
                    if link and self.is_trusted(link):
                        urls.append(link)
                        logger.info(f"[TrustedSearch] ✓ {link}")

                logger.info(f"[TrustedSearch:Google] '{query}' -> {len(urls)}/{len(data['items'])} trusted")
                return urls, False

        except asyncio.TimeoutError:
            logger.error(f"[TrustedSearch:Google] Timeout for '{query}'")
            return [], False
        except Exception as e:
            logger.error(f"[TrustedSearch:Google] Failed for '{query}': {e}")
            return [], False

    # ---------------------------------------------------------------------
    # Single Query Search (Serper.dev fallback)
    # ---------------------------------------------------------------------
    async def search_query_serper(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """
        Runs a single Serper.dev search request and returns trusted URLs.
        Serper provides 2,500 free searches/month.
        """
        if not self.SERPER_API_KEY:
            return []

        headers = {
            "X-API-KEY": self.SERPER_API_KEY,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "num": 10,  # Get 10 results
        }

        try:
            async with session.post(
                self.SERPER_URL,
                headers=headers,
                json=payload,
                timeout=GOOGLE_CSE_TIMEOUT,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[TrustedSearch:Serper] HTTP {resp.status}: {error_text[:200]}")
                    return []

                data = await resp.json()

                # Serper returns results in "organic" array
                organic = data.get("organic", [])
                if not organic:
                    logger.warning(f"[TrustedSearch:Serper] No results for '{query}'")
                    return []

                all_urls = [r.get("link", "N/A") for r in organic]
                logger.info(f"[TrustedSearch:Serper] '{query}' raw: {all_urls[:3]}...")

                urls = []
                for result in organic:
                    link = result.get("link")
                    if link and self.is_trusted(link):
                        urls.append(link)
                        logger.info(f"[TrustedSearch] ✓ {link}")

                logger.info(f"[TrustedSearch:Serper] '{query}' -> {len(urls)}/{len(organic)} trusted")
                return urls

        except asyncio.TimeoutError:
            logger.error(f"[TrustedSearch:Serper] Timeout for '{query}'")
            return []
        except Exception as e:
            logger.error(f"[TrustedSearch:Serper] Failed for '{query}': {e}")
            return []

    # ---------------------------------------------------------------------
    # Unified Search (with fallback)
    # ---------------------------------------------------------------------
    async def search_query(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """
        Search with automatic fallback: Google CSE -> Serper.dev
        """
        # If Google quota already exceeded, go straight to Serper
        if self.google_quota_exceeded:
            if self.serper_available:
                return await self.search_query_serper(session, query)
            return []

        # Try Google first
        if self.google_available:
            urls, quota_exceeded = await self.search_query_google(session, query)

            if quota_exceeded:
                self.google_quota_exceeded = True
                logger.warning("[TrustedSearch] Google CSE quota exceeded! " "Switching to Serper.dev fallback...")

                if self.serper_available:
                    return await self.search_query_serper(session, query)
                else:
                    logger.error(
                        "[TrustedSearch] No SERPER_API_KEY configured. "
                        "Set SERPER_API_KEY env var for fallback search."
                    )
                    return []

            return urls

        # No Google, try Serper directly
        if self.serper_available:
            return await self.search_query_serper(session, query)

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
        Generate queries with site: operator for TOP trusted domains only.
        Reduced to 3 sites for speed - NIH/WHO/PubMed cover most medical facts.
        """
        # Only top 3 most comprehensive medical sources
        priority_sites = [
            "nih.gov",
            "who.int",
            "pubmed.ncbi.nlm.nih.gov",
        ]

        site_queries = []
        for site in priority_sites:
            site_queries.append(f"site:{site} {base_query}")

        return site_queries

    # ---------------------------------------------------------------------
    # Generate All Queries (without executing search)
    # ---------------------------------------------------------------------
    async def generate_search_queries(
        self,
        post_text: str,
        failed_entities: List[str],
        max_queries: int = 6,
    ) -> List[str]:
        """
        Generate all search queries upfront without executing any search API calls.
        This allows the pipeline to control query execution one-by-one for quota optimization.

        Returns:
            List of search queries (LLM-reformulated + site-specific)
        """
        # 1) Generate reformulated queries (1 LLM call)
        queries = await self.reformulate_queries(post_text, failed_entities)
        logger.info(f"[TrustedSearch] Reformulated queries: {queries}")

        if not queries:
            # Fallback to basic query
            queries = [post_text[:100]]

        # 2) Generate site-specific queries for better targeting
        site_queries = self._generate_site_queries(queries[0])

        # 3) Interleave: original, site-specific, original, site-specific...
        all_queries = []
        for i, q in enumerate(queries):
            all_queries.append(q)
            if i < len(site_queries):
                all_queries.append(site_queries[i])
        # Add remaining site queries
        all_queries.extend(site_queries[len(queries) :])

        # Limit total queries
        all_queries = all_queries[:max_queries]
        logger.info(f"[TrustedSearch] Generated {len(all_queries)} queries for quota-optimized search")

        return all_queries

    # ---------------------------------------------------------------------
    # Execute Single Query (quota-optimized)
    # ---------------------------------------------------------------------
    async def execute_single_query(self, query: str) -> List[str]:
        """
        Execute a single search query and return trusted URLs.
        This is the quota-optimized method - call only when needed.

        Returns:
            List of trusted URLs from this single query
        """
        logger.info(f"[TrustedSearch] Executing single query: '{query}'")

        async with aiohttp.ClientSession() as session:
            try:
                urls = await self.search_query(session, query)
                logger.info(f"[TrustedSearch] Single query returned {len(urls)} trusted URLs")
                return urls
            except Exception as e:
                logger.error(f"[TrustedSearch] Single query failed: {e}")
                return []

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Execute a search query and return results formatted as dicts with 'url' key.
        Used by VerdictGenerator for web evidence fetching.

        Returns:
            List of dicts: [{"url": "https://example.com"}, ...]
        """
        urls = await self.execute_single_query(query)
        # Limit results and format as expected by caller
        return [{"url": url} for url in urls[:max_results]]

    # ---------------------------------------------------------------------
    # Main Search Handler (legacy - runs all queries)
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
