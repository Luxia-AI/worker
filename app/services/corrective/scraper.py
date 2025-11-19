import asyncio
from typing import Any, Dict, List

import aiohttp
import trafilatura

from app.core.logger import get_logger

logger = get_logger(__name__)


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

    # ---------------------------------------------------------------------
    # Primary HTTP Fetcher
    # ---------------------------------------------------------------------
    async def fetch_html(self, session: aiohttp.ClientSession, url: str) -> str | None:
        try:
            async with session.get(url, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.warning(f"[Scraper] Non-200 status: {url} — {resp.status}")
                    return None

                html = await resp.text()
                return html

        except Exception as e:
            logger.error(f"[Scraper] HTTP fetch failed for {url}: {e}")
            return None

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
                page = await browser.new_page()

                await page.goto(url, timeout=self.playwright_timeout)
                await page.wait_for_load_state("domcontentloaded")

                html: str = await page.content()
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
        html = await self.fetch_html(session, url)

        # If HTML missing → Playwright fallback
        if html is None:
            logger.info(f"[Scraper] Falling back to Playwright: {url}")
            html = await self.playwright_fallback(url)

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
