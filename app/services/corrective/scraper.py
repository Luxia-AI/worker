import asyncio
import random
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

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
    "Accept-Encoding": "gzip, deflate",
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
        who_playwright_timeout: int = 20000,
        max_line_size: int = 16384,
        max_field_size: int = 65536,
    ):
        self.timeout = timeout
        self.playwright_timeout = playwright_timeout
        self.who_playwright_timeout = who_playwright_timeout
        self.max_line_size = max_line_size
        self.max_field_size = max_field_size
        self.domain_http_timeouts: Dict[str, int] = {
            "canada.ca": 8,
            "jamanetwork.com": 8,
            "cochranelibrary.com": 10,
            "who.int": 14,
        }
        self.domain_playwright_timeouts: Dict[str, int] = {
            "who.int": self.who_playwright_timeout,
            "canada.ca": 5000,
            "jamanetwork.com": 5000,
            "cochranelibrary.com": 6000,
        }
        self.domain_failure_skip_threshold = 2
        self.domain_failure_skip_seconds = 900  # 15 minutes
        self.domain_403_cooldown_seconds = 24 * 60 * 60
        self._domain_failures: Dict[str, Dict[str, float]] = {}
        self._attempted_urls: set[str] = set()
        self._download_event_urls: set[str] = set()

    def _get_headers(self) -> Dict[str, str]:
        """Get browser-like headers with random User-Agent."""
        headers = DEFAULT_HEADERS.copy()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        return headers

    def _extract_domain(self, url: str) -> str | None:
        try:
            host = (urlparse(url).netloc or "").lower()
            if host.startswith("www."):
                host = host[4:]
            return host or None
        except Exception:
            return None

    def _domain_key(self, domain: str | None) -> str | None:
        if not domain:
            return None
        for key in self.domain_http_timeouts:
            if domain == key or domain.endswith(f".{key}"):
                return key
        for key in self.domain_playwright_timeouts:
            if domain == key or domain.endswith(f".{key}"):
                return key
        return domain

    def _http_timeout_for_url(self, url: str) -> int:
        key = self._domain_key(self._extract_domain(url))
        if key and key in self.domain_http_timeouts:
            return int(self.domain_http_timeouts[key])
        return int(self.timeout)

    def _playwright_timeout_for_url(self, url: str, default_timeout: int) -> int:
        key = self._domain_key(self._extract_domain(url))
        if key and key in self.domain_playwright_timeouts:
            return int(self.domain_playwright_timeouts[key])
        return int(default_timeout)

    def _should_skip_domain(self, url: str) -> bool:
        domain = self._extract_domain(url)
        if not domain:
            return False
        state = self._domain_failures.get(domain)
        if not state:
            return False
        blocked_until = float(state.get("blocked_until", 0.0) or 0.0)
        if blocked_until > time.time():
            logger.warning(
                "[Scraper][DomainPolicy] Skipping domain due to recent failures: %s (blocked for %.0fs)",
                domain,
                blocked_until - time.time(),
            )
            return True
        return False

    def _record_domain_failure(self, url: str, stage: str, reason: str) -> None:
        domain = self._extract_domain(url)
        if not domain:
            return
        now = time.time()
        state = self._domain_failures.setdefault(domain, {"count": 0.0, "last": 0.0, "blocked_until": 0.0})
        count = int(state.get("count", 0.0) or 0)
        last = float(state.get("last", 0.0) or 0.0)
        if now - last > self.domain_failure_skip_seconds:
            count = 0
        count += 1
        blocked_until = float(state.get("blocked_until", 0.0) or 0.0)
        if count >= self.domain_failure_skip_threshold:
            blocked_until = now + self.domain_failure_skip_seconds
        state["count"] = float(count)
        state["last"] = now
        state["blocked_until"] = blocked_until
        logger.warning(
            "[Scraper][DomainPolicy] Failure recorded domain=%s stage=%s count=%d reason=%s",
            domain,
            stage,
            count,
            reason[:120],
        )

    def _record_domain_success(self, url: str) -> None:
        domain = self._extract_domain(url)
        if not domain:
            return
        if domain in self._domain_failures:
            self._domain_failures.pop(domain, None)

    def reset_job_attempts(self) -> None:
        self._attempted_urls.clear()
        self._download_event_urls.clear()

    def _mark_attempted(self, url: str) -> bool:
        if url in self._attempted_urls:
            return False
        self._attempted_urls.add(url)
        return True

    def _block_domain_for_cooldown(self, url: str, seconds: int, reason: str) -> None:
        domain = self._extract_domain(url)
        if not domain:
            return
        state = self._domain_failures.setdefault(domain, {"count": 0.0, "last": 0.0, "blocked_until": 0.0})
        state["blocked_until"] = time.time() + seconds
        state["last"] = time.time()
        logger.warning(
            "[Scraper][DomainPolicy] Domain blocked: %s for %.0fs reason=%s",
            domain,
            float(seconds),
            reason,
        )

    @staticmethod
    def _is_download_event_error(error_text: str) -> bool:
        low = (error_text or "").lower()
        return "download is starting" in low

    # ---------------------------------------------------------------------
    # Primary HTTP Fetcher
    # ---------------------------------------------------------------------
    @throttled(limit=30, period=60.0, name="web_scraper")
    async def fetch_html(self, session: aiohttp.ClientSession, url: str) -> tuple[str | None, bool, int | None]:
        """
        Fetch HTML content from URL.

        Returns:
            tuple: (html_content, should_fallback, status_code)
            - html_content: The HTML string or None if fetch failed
            - should_fallback: True if Playwright fallback might help, False if not
            - status_code: HTTP status when available
        """
        try:
            if self._should_skip_domain(url):
                return None, False, None

            headers = self._get_headers()
            http_timeout = self._http_timeout_for_url(url)
            async with session.get(url, timeout=http_timeout, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning(f"[Scraper] Non-200 status: {url} ? {resp.status}")
                    should_fallback = resp.status == 403 or resp.status >= 500
                    self._record_domain_failure(url, stage="http_status", reason=f"status={resp.status}")
                    return None, should_fallback, int(resp.status)

                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                    pdf_bytes = await resp.read()
                    pdf_text = self._extract_pdf_text(pdf_bytes)
                    if pdf_text:
                        self._record_domain_success(url)
                        return f"[[PDF_TEXT]]{pdf_text}", False, int(resp.status)
                    self._record_domain_failure(url, stage="pdf_extract", reason="empty_pdf_text")
                    return None, False, int(resp.status)

                html = await resp.text(errors="ignore")
                if html:
                    self._record_domain_success(url)
                return html, True, int(resp.status)

        except Exception as e:
            logger.error(f"[Scraper] HTTP fetch failed for {url}: {e}")
            self._record_domain_failure(url, stage="http_exception", reason=str(e))
            return None, True, None

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

    def _extract_pdf_text(self, pdf_bytes: bytes) -> str | None:
        try:
            from io import BytesIO

            import pdfplumber

            pages: List[str] = []
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages[:8]:
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(text.strip())
            joined = "\n".join(pages).strip()
            return joined or None
        except Exception as e:
            logger.warning(f"[Scraper] PDF extraction failed: {e}")
            return None

    # ---------------------------------------------------------------------
    # Playwright Fallback (JS-rendered pages)
    # ---------------------------------------------------------------------
    async def playwright_fallback(self, url: str, timeout_override: int | None = None) -> str | None:
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

                timeout = timeout_override or self.playwright_timeout
                timeout = self._playwright_timeout_for_url(url, timeout)
                await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                await page.wait_for_load_state("domcontentloaded")

                html: str = await page.content()
                await context.close()
                await browser.close()
                if html:
                    self._record_domain_success(url)
                return html

        except Exception as e:
            logger.error(f"[Scraper] Playwright fallback failed for {url}: {e}")
            err = str(e)
            if self._is_download_event_error(err):
                # Download events are not access denials; do not poison domain health.
                logger.info("[Scraper][DomainPolicy] Playwright download event observed for %s; no domain penalty", url)
                self._download_event_urls.add(url)
            else:
                self._record_domain_failure(url, stage="playwright_exception", reason=err)
            return None

    # ---------------------------------------------------------------------
    # Scrape a single URL
    # ---------------------------------------------------------------------
    async def scrape_one(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        logger.info(f"[Scraper] Scraping URL: {url}")
        if not self._mark_attempted(url):
            logger.info(f"[Scraper] Skipping duplicate URL attempted in this job: {url}")
            return {"url": url, "content": None, "source": None, "published_at": None}
        if self._should_skip_domain(url):
            logger.warning(f"[Scraper] Skipped by domain policy: {url}")
            return {"url": url, "content": None, "source": None, "published_at": None}
        is_who = "who.int" in url.lower()

        html, should_fallback, status_code = await self.fetch_html(session, url)

        if html is None and should_fallback:
            logger.info(f"[Scraper] Falling back to Playwright: {url}")
            timeout_override = self.who_playwright_timeout if is_who else max(self.playwright_timeout, 12000)
            html = await self.playwright_fallback(url, timeout_override=timeout_override)
        elif html is None:
            logger.info(f"[Scraper] Skipping Playwright fallback (client error): {url}")

        if (
            html is None
            and status_code == 403
            and url not in self._download_event_urls
            and not self._should_skip_domain(url)
        ):
            self._block_domain_for_cooldown(url, self.domain_403_cooldown_seconds, reason="persistent_403")

        if not isinstance(html, str) or not html:
            logger.warning(f"[Scraper] No HTML content for {url}")
            self._record_domain_failure(url, stage="empty_html", reason="no_html_content")
            return {"url": url, "content": None, "source": None, "published_at": None}

        if html.startswith("[[PDF_TEXT]]"):
            text_content = html[len("[[PDF_TEXT]]") :].strip()
        else:
            text_content = self.extract_text(html)

        if not text_content:
            logger.warning(f"[Scraper] No extractable text for {url}")
            self._record_domain_failure(url, stage="extract_text", reason="empty_extracted_text")
            return {"url": url, "content": None, "source": None, "published_at": None}

        source = self._extract_domain(url)
        published_at = self._extract_publish_date(html)

        logger.info(f"[Scraper] Extracted {len(text_content)} chars from {url}")
        self._record_domain_success(url)

        return {
            "url": url,
            "content": text_content,
            "source": source,
            "published_at": published_at,
        }

    # ---------------------------------------------------------------------
    # Scrape multiple URLs in parallel
    # ---------------------------------------------------------------------
    async def scrape_all(self, urls: List[str], reset_attempts: bool = False) -> List[Dict[str, Any]]:
        if reset_attempts:
            self.reset_job_attempts()
        logger.info(f"[Scraper] Starting scrape for {len(urls)} URLs")

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            max_line_size=self.max_line_size,
            max_field_size=self.max_field_size,
        ) as session:
            tasks = [self.scrape_one(session, url) for url in urls]
            results = await asyncio.gather(*tasks)

        return results

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
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
