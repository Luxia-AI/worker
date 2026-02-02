import asyncio
import random
from typing import Any, Dict, List

import aiohttp
import trafilatura

from app.core.logger import get_logger
from app.core.rate_limit import throttled

logger = get_logger(__name__)

# Browser-like User-Agent strings to avoid 403 blocks from academic sites
USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) " "Gecko/20100101 Firefox/121.0"),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
    ),
]

# Standard browser headers to mimic real browser requests
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


class Scraper:
    """
    Asynchronous scraper:
        1. Try Trafilatura (fast, HTML → text)
        2. If failure or empty → fallback to Playwright JS-render
        3. Return clean extracted text
    """

    def __init__(
        self,
        timeout: int = 12,
        playwright_timeout: int = 8000,
    ):
        self.timeout = timeout
        self.playwright_timeout = playwright_timeout

    def _get_headers(self) -> Dict[str, str]:
        """Get browser-like headers with random User-Agent."""
        headers = DEFAULT_HEADERS.copy()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        return headers

    # ---------------------------------------------------------------------
    # Primary HTTP Fetcher
    # ---------------------------------------------------------------------
    @throttled(limit=30, period=60.0, name="web_scraper")
    async def fetch_html(self, session: aiohttp.ClientSession, url: str) -> tuple[str | None, bool]:
        """
        Fetch HTML content from URL.

        Returns:
            tuple: (html_content, should_fallback)
            - html_content: The HTML string or None if fetch failed
            - should_fallback: True if Playwright fallback might help, False if not
              (e.g., 403/404 errors should NOT trigger fallback)
        """
        try:
            headers = self._get_headers()
            async with session.get(url, timeout=self.timeout, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning(f"[Scraper] Non-200 status: {url} — {resp.status}")
                    # 4xx client errors (403, 404, etc.) won't be fixed by Playwright
                    # Only network issues or empty content should trigger fallback
                    should_fallback = resp.status >= 500  # Only server errors
                    return None, should_fallback

                html = await resp.text()
                return html, True  # Success, but fallback OK if extraction fails

        except Exception as e:
            logger.error(f"[Scraper] HTTP fetch failed for {url}: {e}")
            return None, True  # Network error - Playwright might help

    # ---------------------------------------------------------------------
    # Trafilatura Extraction
    # ---------------------------------------------------------------------
    def extract_text(self, html: str) -> str | None:
        try:
            text = trafilatura.extract(html, include_comments=False)
            if text is not None:
                result: str = str(text).strip()
                return result
            return None
        except Exception as e:
            logger.error(f"[Scraper] Trafilatura extraction failed: {e}")
            return None

    # ---------------------------------------------------------------------
    # Playwright Fallback (JS-rendered pages)
    # ---------------------------------------------------------------------
    async def playwright_fallback(self, url: str) -> str | None:
        """
        Only used when Trafilatura fails or HTML is empty.
        Requires Playwright installed with: playwright install chromium
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error("[Scraper] Playwright not installed — skipping fallback.")
            return None

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                # Set browser-like context to avoid detection
                context = await browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                )
                page = await context.new_page()

                await page.goto(url, timeout=self.playwright_timeout)
                await page.wait_for_load_state("domcontentloaded")

                html: str = await page.content()
                await context.close()
                await browser.close()
                return html

        except Exception as e:
            logger.error(f"[Scraper] Playwright fallback failed for {url}: {e}")
            return None

    # ---------------------------------------------------------------------
    # Scrape a single URL
    # ---------------------------------------------------------------------
    async def scrape_one(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        logger.info(f"[Scraper] Scraping URL: {url}")

        # Step 1: Try normal HTTP fetch
        html, should_fallback = await self.fetch_html(session, url)

        # If HTML missing and fallback is appropriate → Playwright fallback
        # Do NOT fallback for 4xx errors (403, 404, etc.) - they won't be fixed by JS rendering
        if html is None and should_fallback:
            logger.info(f"[Scraper] Falling back to Playwright: {url}")
            html = await self.playwright_fallback(url)
        elif html is None:
            logger.info(f"[Scraper] Skipping Playwright fallback (client error): {url}")

        if not isinstance(html, str) or not html:
            logger.warning(f"[Scraper] No HTML content for {url}")
            return {"url": url, "content": None, "source": None, "published_at": None}

        # Step 2: Extract text (Trafilatura)
        text = self.extract_text(html)

        if not text:
            logger.warning(f"[Scraper] No extractable text for {url}")
            return {"url": url, "content": None, "source": None, "published_at": None}

        # Step 3: Metadata parsing
        source = self._extract_domain(url)
        published_at = self._extract_publish_date(html)

        logger.info(f"[Scraper] Extracted {len(text)} chars from {url}")

        return {
            "url": url,
            "content": text,
            "source": source,
            "published_at": published_at,
        }

    # ---------------------------------------------------------------------
    # Scrape multiple URLs in parallel
    # ---------------------------------------------------------------------
    async def scrape_all(self, urls: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"[Scraper] Starting scrape for {len(urls)} URLs")

        async with aiohttp.ClientSession() as session:
            tasks = [self.scrape_one(session, url) for url in urls]
            results = await asyncio.gather(*tasks)

        return results

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _extract_domain(url: str) -> str | None:
        try:
            return url.split("/")[2]
        except Exception:
            return None

    @staticmethod
    def _extract_publish_date(html: str) -> str | None:
        """
        Simple heuristic: search for <meta> tags containing dates.
        More advanced logic can be added later.
        """
        import re
        from datetime import datetime

        # Common patterns in medical articles
        patterns = [
            r'"datePublished":"(.*?)"',
            r'"dateModified":"(.*?)"',
            r"<meta property=\"article:published_time\" content=\"(.*?)\"",
        ]

        for p in patterns:
            match = re.search(p, html)
            if match:
                try:
                    dt = datetime.fromisoformat(match.group(1).replace("Z", ""))
                    return dt.isoformat()
                except Exception:  # nosec B112
                    continue

        return None
