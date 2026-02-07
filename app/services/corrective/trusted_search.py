import asyncio
import re
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
from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy

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
    # Deterministic Query Helpers
    # ---------------------------------------------------------------------
    _STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "just",
        "like",
        "every",
        "individual",
        "around",
        "to",
        "for",
        "of",
        "in",
        "on",
        "with",
        "by",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "about",
        "around",
        "average",
        "per",
        "times",
        "every",
        "each",
        "roughly",
        "approximately",
    }
    _ACTION_WORDS = {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "how",
        "what",
        "when",
        "where",
        "why",
        "which",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
    }
    _CONSTRAINT_KEYWORDS = [
        "statistics",
        "prevalence",
        "incidence",
        "guideline",
        "systematic review",
        "meta-analysis",
    ]
    _MERGE_PREFIXES = ("and ", "or ", "but ", "however ", "although ", "while ")

    def merge_subclaims(self, subclaims: List[str]) -> List[str]:
        """Merge short or conjunction-leading subclaims into the previous segment."""
        merged: List[str] = []
        for raw in subclaims or []:
            s = (raw or "").strip()
            if not s:
                continue
            if not merged:
                merged.append(s)
                continue
            lower = s.lower()
            short = len(s) < 40 or len(s.split()) < 6
            if lower.startswith(self._MERGE_PREFIXES) or short:
                merged[-1] = f"{merged[-1].rstrip(', ')} {s}".strip()
            else:
                merged.append(s)
        return merged

    def _extract_numbers(self, text: str) -> List[str]:
        if not text:
            return []
        nums = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)
        cleaned = []
        for n in nums:
            n = n.replace(",", "")
            if n and n not in cleaned:
                cleaned.append(n)
        return cleaned

    def _extract_numbers_raw(self, text: str) -> List[str]:
        if not text:
            return []
        nums = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)
        cleaned = []
        for n in nums:
            if n and n not in cleaned:
                cleaned.append(n)
        return cleaned

    def _tokenize_words(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", text.lower())

    def _collect_key_terms(self, texts: List[str], entities: List[str] | None = None) -> tuple[list[str], list[str]]:
        entities = entities or []
        key_terms: List[str] = []
        key_numbers: List[str] = []

        for t in texts:
            for n in self._extract_numbers(t):
                if n not in key_numbers:
                    key_numbers.append(n)
            for w in self._tokenize_words(t):
                if w in self._STOPWORDS:
                    continue
                if w not in key_terms:
                    key_terms.append(w)

        for e in entities:
            for w in self._tokenize_words(e):
                if w in self._STOPWORDS:
                    continue
                if w not in key_terms:
                    key_terms.append(w)

        return key_terms, key_numbers

    def _extract_phrase_candidates(self, text: str, entities: List[str] | None = None) -> List[str]:
        entities = entities or []
        tokens = [t for t in self._tokenize_words(text) if t not in self._STOPWORDS and t not in self._ACTION_WORDS]
        phrases: List[str] = []

        # Add non-trivial entities as exact phrases.
        for ent in entities:
            ent_clean = " ".join(self._tokenize_words(ent))
            if len(ent_clean.split()) >= 2 and ent_clean not in phrases:
                phrases.append(ent_clean)

        # Backfill with token bigrams (noun-focused approximation).
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i + 1]}"
            if bg not in phrases and len(bg) >= 8:
                phrases.append(bg)
            if len(phrases) >= 4:
                break

        return phrases[:4]

    def _build_phrase_variants(self, phrase: str) -> List[str]:
        """
        Build lightweight lexical variants from an input phrase without
        domain- or claim-specific hardcoded synonym dictionaries.
        """
        norm = " ".join(self._tokenize_words(phrase))
        if not norm:
            return []

        words = norm.split()
        variants = [norm]

        # Singular/plural variant on last token.
        # Keep conservative to avoid malformed forms like "diet richs".
        if words:
            last = words[-1]
            if len(words) == 1 and last.endswith("s") and len(last) > 3:
                singular = " ".join(words[:-1] + [last[:-1]])
                if singular not in variants:
                    variants.append(singular)
            elif len(words) == 1 and not last.endswith("s"):
                plural = " ".join(words[:-1] + [last + "s"])
                if plural not in variants:
                    variants.append(plural)

        # Hyphen/space normalization variant.
        if "-" in norm:
            dehyphen = norm.replace("-", " ")
            if dehyphen not in variants:
                variants.append(dehyphen)

        # Keep unique, short list for query compactness.
        out: List[str] = []
        for v in variants:
            if v not in out:
                out.append(v)
            if len(out) >= 3:
                break
        return out

    def _build_boolean_synonym_block(self, text: str, entities: List[str] | None = None) -> str:
        entities = entities or []
        phrases = self._extract_phrase_candidates(text, entities=entities)
        if not phrases:
            return ""
        best_phrase = phrases[0]
        variants = self._build_phrase_variants(best_phrase)
        if len(variants) < 2:
            return ""
        quoted = [f'"{term}"' for term in variants]
        return "(" + " OR ".join(quoted) + ")"

    def _build_negative_filters(self, text: str) -> List[str]:
        t = (text or "").lower()
        negatives: List[str] = []
        if "adult" in t or "adults" in t:
            negatives.extend(["-fetal", "-pediatric"])
        if "human" in t:
            negatives.extend(["-veterinary"])
        return negatives

    def _build_research_instruction_query(self, text: str, entities: List[str] | None = None) -> str:
        entities = entities or []
        phrases = self._extract_phrase_candidates(text, entities=entities)
        bool_block = self._build_boolean_synonym_block(text, entities=entities)
        negatives = self._build_negative_filters(text)
        numbers = self._extract_numbers_raw(text)[:2]

        parts: List[str] = []
        if bool_block:
            parts.append(bool_block)
        for p in phrases[:2]:
            parts.append(f'"{p}"')
        for n in numbers:
            parts.append(n)

        # Add one strong research/data constraint to reduce noisy results.
        text_lower = (text or "").lower()
        chosen_constraint = "statistics"
        for c in self._CONSTRAINT_KEYWORDS:
            if c in text_lower:
                chosen_constraint = c
                break
        parts.append(chosen_constraint)

        parts.extend(negatives[:2])
        return " ".join(parts).strip()

    def _build_direct_query(self, text: str, entities: List[str] | None = None) -> str:
        if not text:
            return ""
        entities = entities or []
        tokens = [t for t in self._tokenize_words(text) if t not in self._STOPWORDS]
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        entity_tokens = set(self._tokenize_words(" ".join(entities)))

        def _score(tok: str) -> float:
            score = 0.0
            score += 2.0 * freq.get(tok, 0)
            score += min(1.0, len(tok) / 8.0)
            if tok in entity_tokens:
                score += 1.0
            return score

        ranked_terms = sorted(set(tokens), key=lambda t: (-_score(t), t))
        numbers_raw = self._extract_numbers_raw(text)
        numbers_norm = self._extract_numbers(text)
        # Avoid number-only queries; keep at most 3 numbers
        numbers_raw = numbers_raw[:3]
        numbers_norm = numbers_norm[:3]

        terms: List[str] = []
        for n in numbers_raw:
            terms.append(n)
        # include normalized number if raw had commas stripped
        for n in numbers_norm:
            if n not in terms:
                terms.append(n)
        for w in ranked_terms:
            terms.append(w)

        # Keep query concise and ensure at least 2 content words
        terms = dedupe_list(terms)
        # Ensure at least two alphabetic tokens are retained
        word_terms = [t for t in terms if t.isalpha()]
        if len(word_terms) < 2:
            for w in ranked_terms:
                if w not in terms:
                    terms.append(w)
                if len([t for t in terms if t.isalpha()]) >= 2:
                    break
        terms = terms[:7]
        return " ".join(terms).strip()

    def _build_direct_queries(self, subclaims: List[str], entities: List[str] | None = None) -> List[str]:
        entities = entities or []
        queries: List[str] = []
        for sub in subclaims:
            q = self._build_direct_query(sub, entities=entities)
            if q:
                queries.append(q)
        return dedupe_list(queries)

    def _query_has_key_terms(self, query: str, key_terms: List[str], key_numbers: List[str]) -> bool:
        if not query:
            return False
        q = query.lower()
        q_num = q.replace(",", "")
        for n in key_numbers:
            if n and n in q_num:
                return True
        for term in key_terms:
            if term and re.search(rf"\b{re.escape(term)}\b", q):
                return True
        return False

    def _filter_queries(self, queries: List[str], key_terms: List[str], key_numbers: List[str]) -> List[str]:
        filtered: List[str] = []
        for q in queries:
            if not isinstance(q, str):
                continue
            q = q.strip().lower()
            if not q:
                continue
            if not self._query_has_key_terms(q, key_terms, key_numbers):
                continue
            filtered.append(q)
        return dedupe_list(filtered)

    # ---------------------------------------------------------------------
    # Query Reformulation
    # ---------------------------------------------------------------------
    def _build_question_queries(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = set(self._tokenize_words(text))
        queries: List[str] = []

        if "bones" in tokens or "bone" in tokens:
            if "hands" in tokens or "hand" in tokens or "feet" in tokens or "foot" in tokens:
                queries.append("how many bones are in adult human hands and feet")
                queries.append("how many bones are in the hands and feet")
            if "adult" in tokens or "adults" in tokens:
                queries.append("how many bones does an adult human have")
            else:
                queries.append("how many bones does a human have")

        if "heart" in tokens and ("beats" in tokens or "beat" in tokens):
            if "day" in tokens or "daily" in tokens:
                queries.append("how many times does the human heart beat per day")

        if "cells" in tokens or "cell" in tokens:
            if "second" in tokens or "seconds" in tokens:
                queries.append("how many new cells does the human body produce per second")

        if "tongue" in tokens and ("print" in tokens or "prints" in tokens):
            queries.append("are tongue prints unique")
            queries.append("unique tongue prints individuals")

        return dedupe_list(queries)

    def _build_advanced_operator_queries(self, subclaim: str, entities: List[str] | None = None) -> List[str]:
        entities = entities or []
        base = self._build_research_instruction_query(subclaim, entities=entities)
        if not base:
            return []

        adv = [base]
        # Research-paper style query.
        adv.append(f"{base} filetype:pdf")
        # Strong title constraint variant.
        adv.append(f'intitle:"meta-analysis" {base}')
        return dedupe_list(adv)

    async def reformulate_queries(
        self,
        text: str,
        failed_entities: List[str],
        entities: List[str] | None = None,
        subclaims: List[str] | None = None,
    ) -> List[str]:
        """
        LLM-powered search query reformulation.
        Produces highly optimized evidence retrieval phrases.
        """
        entities = entities or []
        subclaims = subclaims or []

        direct_query = self._build_direct_query(text, entities=entities)
        key_terms, key_numbers = self._collect_key_terms(texts=[text] + subclaims, entities=entities)

        prompt = f"""
{QUERY_REFORMULATION_PROMPT}

POST TEXT:
{text}

SUBCLAIMS:
{subclaims}

ENTITIES:
{entities}

FAILED ENTITIES:
{failed_entities}
"""

        try:
            # HIGH priority: Query reformulation is crucial for good search results
            result = await self.llm_client.ainvoke(prompt, response_format="json", priority=LLMPriority.HIGH)
            queries = result.get("queries", [])
            cleaned = [q.strip().lower() for q in queries if isinstance(q, str)]
            merged = [direct_query] + cleaned if direct_query else cleaned
            filtered = self._filter_queries(merged, key_terms, key_numbers)
            return dedupe_list(filtered)  # dedupe but preserve order

        except Exception as e:
            logger.error(f"[TrustedSearch] LLM query reformulation failed: {e}")
            # fallback -> old heuristic (plus direct query)
            fallback = self._fallback_queries(text, failed_entities)
            merged = [direct_query] + fallback if direct_query else fallback
            filtered = self._filter_queries(merged, key_terms, key_numbers)
            return dedupe_list(filtered)

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
    def _generate_site_queries(self, base_query: str, max_sites: int = 4) -> List[str]:
        """
        Generate queries with site: operator for TOP trusted domains only.
        Kept small for speed and quota efficiency.
        """
        # Prioritize high-yield medical sources (capped to 3-4 sites)
        priority_sites = [
            "medlineplus.gov",
            "cdc.gov",
            "nih.gov",
            "who.int",
            "pubmed.ncbi.nlm.nih.gov",
        ][:max_sites]

        site_queries = []
        for site in priority_sites:
            if base_query:
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
        subclaims: List[str] | None = None,
        entities: List[str] | None = None,
    ) -> List[str]:
        """
        Generate all search queries upfront without executing any search API calls.
        This allows the pipeline to control query execution one-by-one for quota optimization.

        Returns:
            List of search queries (direct subclaim + site-specific + LLM-reformulated)
        """
        entities = entities or []

        # 1) Determine subclaims and merge fragments for better query quality
        if subclaims is None:
            subclaims = AdaptiveTrustPolicy().decompose_claim(post_text)
        merged_subclaims = self.merge_subclaims(subclaims or [])

        key_terms, key_numbers = self._collect_key_terms(texts=[post_text] + merged_subclaims, entities=entities)

        # 2) Build deterministic direct + operator-rich research queries per subclaim (highest priority)
        direct_queries = []
        advanced_queries = []
        question_queries = []
        for sub in merged_subclaims:
            q = self._build_direct_query(sub, entities=entities)
            if q:
                direct_queries.append(q)
            advanced_queries.extend(self._build_advanced_operator_queries(sub, entities=entities))
            q_questions = self._build_question_queries(sub)
            if q_questions:
                question_queries.append(q_questions[0])  # keep 1 best question per subclaim

        direct_queries = self._filter_queries(direct_queries, key_terms, key_numbers)
        advanced_queries = self._filter_queries(advanced_queries, key_terms, key_numbers)
        question_queries = self._filter_queries(question_queries, key_terms, key_numbers)

        # 3) Site-specific variants for top direct/operator queries
        site_queries: List[str] = []
        for q in direct_queries[:1] + advanced_queries[:1]:
            site_queries.extend(self._generate_site_queries(q))

        all_queries = dedupe_list(advanced_queries + direct_queries + question_queries + site_queries)

        # 4) If budget remains, add LLM reformulated queries
        if len(all_queries) < max_queries:
            llm_queries = await self.reformulate_queries(
                post_text,
                failed_entities,
                entities=entities,
                subclaims=merged_subclaims,
            )
            llm_queries = self._filter_queries(llm_queries, key_terms, key_numbers)
            for q in llm_queries:
                if q not in all_queries:
                    all_queries.append(q)

        # Final validation (strictly direct to claim terms) and budget cap
        all_queries = self._filter_queries(all_queries, key_terms, key_numbers)
        all_queries = dedupe_list(all_queries)[:max_queries]
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
        Uses LLM (Groq) to generate highly optimized reinforcement search queries.
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
