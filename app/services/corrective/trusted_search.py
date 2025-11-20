import asyncio
import json
from typing import Any, Dict, List

import aiohttp

from app.constants.config import (
    GOOGLE_CSE_SEARCH_URL,
    GOOGLE_CSE_TIMEOUT,
    LLM_MAX_TOKENS_REINFORCEMENT,
    LLM_TEMPERATURE,
    TRUSTED_DOMAINS,
)
from app.constants.llm_prompts import QUERY_REFORMULATION_PROMPT, REINFORCEMENT_QUERY_PROMPT
from app.core.config import settings
from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService

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
            async with session.get(url, timeout=GOOGLE_CSE_TIMEOUT) as resp:
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

    async def google_search(self, post_text: str, failed_entities: List[str] = None) -> List[str]:
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
        Uses Groq LLM to generate highly optimized reinforcement search queries.
        Much better than heuristic string concatenation.
        """

        base_statements = [item.get("statement", "") for item in low_conf_items]
        base_entities = failed_entities or []

        prompt = REINFORCEMENT_QUERY_PROMPT.format(statements=base_statements, entities=base_entities)

        try:
            resp = self.groq_client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS_REINFORCEMENT,
            )

            raw = resp.choices[0].message.content
            queries = json.loads(raw)

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

            return list(dict.fromkeys(fallback))

    # ---------------------------------------------------------------------
    # Reinforced Search for Low Confidence Cases
    # ---------------------------------------------------------------------
    async def reinforce_search(
        self,
        low_conf_items: List[Dict],
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
        return list(dict.fromkeys(all_urls))
