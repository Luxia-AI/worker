"""
WHO/IRIS/IMSEAR PDF + HTML Scraper

A robust scraper for WHO, IRIS, and IMSEAR websites that avoids 400s and Playwright timeouts.
"""

import re
import time
from enum import Enum
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx
from playwright.async_api import Browser, Playwright, async_playwright
from pydantic import BaseModel

from app.core.logger import get_logger

logger = get_logger(__name__)


class URLType(Enum):
    """URL classification types."""

    PDF_DIRECT = "pdf_direct"
    WHO_API_BITSTREAM = "who_api_bitstream"
    IRIS_HTML = "iris_html"
    IMSEAR_DOWNLOAD = "imsear_download"
    GENERIC_HTML = "generic_html"


class ScrapeResult(BaseModel):
    """Structured scrape result."""

    url: str
    status: str  # "success", "error", "timeout", etc.
    content_type: str  # "pdf", "html", "text"
    chars_extracted: int
    method_used: str  # "http", "playwright", "fallback"
    error_code: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict] = None


class ScraperMetrics:
    """Metrics for scraper performance."""

    def __init__(self):
        self.success_count = 0
        self.total_count = 0
        self.errors_by_code: Dict[str, int] = {}
        self.latency_by_method: Dict[str, list] = {}
        self.success_by_domain: Dict[str, Tuple[int, int]] = {}  # (success, total)

    def record_attempt(self, domain: str, method: str, success: bool, latency: float, error_code: Optional[str] = None):
        """Record a scrape attempt."""
        self.total_count += 1
        if success:
            self.success_count += 1

        # Domain success rate
        if domain not in self.success_by_domain:
            self.success_by_domain[domain] = (0, 0)
        success_count, total_count = self.success_by_domain[domain]
        total_count += 1
        if success:
            success_count += 1
        self.success_by_domain[domain] = (success_count, total_count)

        # Method latency
        if method not in self.latency_by_method:
            self.latency_by_method[method] = []
        self.latency_by_method[method].append(latency)

        # Error codes
        if error_code:
            self.errors_by_code[error_code] = self.errors_by_code.get(error_code, 0) + 1

    def get_success_rate_by_domain(self) -> Dict[str, float]:
        """Get success rate by domain."""
        return {
            domain: success / total if total > 0 else 0.0 for domain, (success, total) in self.success_by_domain.items()
        }

    def get_avg_latency_by_method(self) -> Dict[str, float]:
        """Get average latency by method."""
        return {
            method: sum(latencies) / len(latencies) if latencies else 0.0
            for method, latencies in self.latency_by_method.items()
        }

    def get_error_counts(self) -> Dict[str, int]:
        """Get error counts by code."""
        return self.errors_by_code.copy()


class WHOScraper:
    """Robust WHO/IRIS/IMSEAR scraper."""

    # WHO domains
    WHO_DOMAINS = {"who.int", "iris.who.int", "imsear.searo.who.int"}

    # Headers for HTTP requests
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    # Alternate headers for fallback
    FALLBACK_HEADERS = {
        **DEFAULT_HEADERS,
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Referer": "https://www.google.com/",
    }

    def __init__(self):
        self.metrics = ScraperMetrics()
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    async def __aenter__(self):
        """Initialize Playwright browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--no-first-run",
                "--no-zygote",
                "--disable-gpu",
            ],
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up Playwright."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    def classify_url(self, url: str) -> URLType:
        """Classify URL type."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Check if it's a WHO domain
        is_who_domain = any(domain.endswith(d) for d in self.WHO_DOMAINS)

        if not is_who_domain:
            return URLType.GENERIC_HTML

        # Check for direct PDF
        if url.lower().endswith(".pdf"):
            return URLType.PDF_DIRECT

        # Check for WHO API bitstream
        if "/server/api/core/bitstreams/" in url and "/content" in url:
            return URLType.WHO_API_BITSTREAM

        # Check for IMSEAR download
        if "imsear.searo.who.int" in domain and ("download" in url.lower() or "handle" in url.lower()):
            return URLType.IMSEAR_DOWNLOAD

        # IRIS HTML (landing pages)
        if "iris.who.int" in domain:
            return URLType.IRIS_HTML

        # Default to generic HTML for WHO domains
        return URLType.GENERIC_HTML

    async def scrape(self, url: str) -> ScrapeResult:
        """Main scrape method."""
        start_time = time.time()
        url_type = self.classify_url(url)
        domain = urlparse(url).netloc

        try:
            if url_type in {URLType.PDF_DIRECT, URLType.WHO_API_BITSTREAM, URLType.IMSEAR_DOWNLOAD}:
                # Try HTTP first for PDFs
                result = await self._scrape_http(url, url_type)
                if result.status == "success":
                    latency = time.time() - start_time
                    self.metrics.record_attempt(domain, "http", True, latency)
                    return result

                # Fallback to Playwright if HTTP failed
                result = await self._scrape_playwright(url, url_type)
                latency = time.time() - start_time
                success = result.status == "success"
                self.metrics.record_attempt(domain, "playwright", success, latency, result.error_code)
                return result

            elif url_type in {URLType.IRIS_HTML, URLType.GENERIC_HTML}:
                # Try HTTP first, fallback to Playwright
                result = await self._scrape_http(url, url_type)
                if result.status == "success":
                    latency = time.time() - start_time
                    self.metrics.record_attempt(domain, "http", True, latency)
                    return result

                # Extract PDF link from HTML if it's a landing page
                if url_type == URLType.IRIS_HTML and result.content:
                    pdf_url = self._extract_pdf_url_from_html(result.content, url)
                    if pdf_url:
                        # Recursively scrape the PDF URL
                        return await self.scrape(pdf_url)

                # Fallback to Playwright
                result = await self._scrape_playwright(url, url_type)
                latency = time.time() - start_time
                success = result.status == "success"
                self.metrics.record_attempt(domain, "playwright", success, latency, result.error_code)
                return result

            else:
                # Generic HTML - use Playwright
                result = await self._scrape_playwright(url, url_type)
                latency = time.time() - start_time
                success = result.status == "success"
                self.metrics.record_attempt(domain, "playwright", success, latency, result.error_code)
                return result

        except Exception as e:
            latency = time.time() - start_time
            error_code = f"exception_{type(e).__name__}"
            self.metrics.record_attempt(domain, "unknown", False, latency, error_code)
            logger.error(f"Scrape failed for {url}: {e}")
            return ScrapeResult(
                url=url,
                status="error",
                content_type="unknown",
                chars_extracted=0,
                method_used="unknown",
                error_code=error_code,
                content=None,
            )

    async def _scrape_http(self, url: str, url_type: URLType) -> ScrapeResult:
        """Scrape using HTTP requests."""
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                # Try with default headers
                response = await client.get(url, headers=self.DEFAULT_HEADERS)

                if response.status_code == 200:
                    return await self._process_http_response(response, url, url_type, "http")

                elif response.status_code in {400, 403, 404}:
                    # Try fallback headers
                    response = await client.get(url, headers=self.FALLBACK_HEADERS)
                    if response.status_code == 200:
                        return await self._process_http_response(response, url, url_type, "http_fallback")

                    # Try HEAD then GET
                    head_response = await client.head(url, headers=self.FALLBACK_HEADERS)
                    if head_response.status_code == 200:
                        response = await client.get(url, headers=self.FALLBACK_HEADERS)
                        if response.status_code == 200:
                            return await self._process_http_response(response, url, url_type, "http_head_fallback")

                return ScrapeResult(
                    url=url,
                    status="error",
                    content_type="unknown",
                    chars_extracted=0,
                    method_used="http",
                    error_code=f"http_{response.status_code}",
                    content=None,
                )

        except Exception as e:
            return ScrapeResult(
                url=url,
                status="error",
                content_type="unknown",
                chars_extracted=0,
                method_used="http",
                error_code=f"http_exception_{type(e).__name__}",
                content=None,
            )

    async def _process_http_response(
        self, response: httpx.Response, url: str, url_type: URLType, method: str
    ) -> ScrapeResult:
        """Process HTTP response."""
        content_type = response.headers.get("content-type", "").lower()

        if url_type == URLType.WHO_API_BITSTREAM or "application/pdf" in content_type or url.lower().endswith(".pdf"):
            # Binary PDF
            try:
                import pdfplumber

                with pdfplumber.open(response.content) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                return ScrapeResult(
                    url=url,
                    status="success",
                    content_type="pdf",
                    chars_extracted=len(text),
                    method_used=method,
                    content=text,
                    metadata={"pages": len(pdf.pages)},
                )
            except Exception as e:
                return ScrapeResult(
                    url=url,
                    status="error",
                    content_type="pdf",
                    chars_extracted=0,
                    method_used=method,
                    error_code=f"pdf_extraction_{type(e).__name__}",
                    content=None,
                )

        else:
            # HTML content
            html_content = response.text
            text_content = self._extract_text_from_html(html_content)

            return ScrapeResult(
                url=url,
                status="success",
                content_type="html",
                chars_extracted=len(text_content),
                method_used=method,
                content=text_content,
            )

    async def _scrape_playwright(self, url: str, url_type: URLType) -> ScrapeResult:
        """Scrape using Playwright with budget constraints."""
        if not self._browser:
            return ScrapeResult(
                url=url,
                status="error",
                content_type="unknown",
                chars_extracted=0,
                method_used="playwright",
                error_code="browser_not_initialized",
            )

        try:
            context = await self._browser.new_context(
                user_agent=self.DEFAULT_HEADERS["User-Agent"], viewport={"width": 1280, "height": 720}
            )

            # Block heavy resources
            await context.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.resource_type in {"image", "media", "font", "stylesheet"}
                    else route.continue_()
                ),
            )

            page = await context.new_page()

            # Set timeouts
            page.set_default_timeout(15000)  # 15s navigation timeout

            # Navigate with timeout
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                await context.close()
                return ScrapeResult(
                    url=url,
                    status="timeout",
                    content_type="unknown",
                    chars_extracted=0,
                    method_used="playwright",
                    error_code="navigation_timeout",
                )

            # Extract content based on type
            if url_type in {URLType.PDF_DIRECT, URLType.WHO_API_BITSTREAM, URLType.IMSEAR_DOWNLOAD}:
                # For PDFs, try to get the binary content
                try:
                    # Check if page has PDF content
                    content = await page.content()
                    if "application/pdf" in content or url.lower().endswith(".pdf"):
                        # This is tricky - Playwright doesn't easily give binary PDF
                        # For now, return HTML content
                        text_content = self._extract_text_from_html(content)
                        await context.close()
                        return ScrapeResult(
                            url=url,
                            status="success",
                            content_type="html",
                            chars_extracted=len(text_content),
                            method_used="playwright",
                            content=text_content,
                        )
                except Exception:
                    pass

            # Get HTML content
            content = await page.content()
            text_content = self._extract_text_from_html(content)

            await context.close()

            return ScrapeResult(
                url=url,
                status="success",
                content_type="html",
                chars_extracted=len(text_content),
                method_used="playwright",
                content=text_content,
            )

        except Exception as e:
            return ScrapeResult(
                url=url,
                status="error",
                content_type="unknown",
                chars_extracted=0,
                method_used="playwright",
                error_code=f"playwright_{type(e).__name__}",
            )

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML."""
        try:
            from readability import Document

            doc = Document(html)
            return doc.summary()
        except ImportError:
            # Fallback to trafilatura if readability is not available
            try:
                import trafilatura

                return trafilatura.extract(html) or ""
            except ImportError:
                # Basic fallback
                return re.sub(r"<[^>]+>", "", html).strip()

    def _extract_pdf_url_from_html(self, html: str, base_url: str) -> Optional[str]:
        """Extract PDF URL from HTML content."""
        # Look for common PDF link patterns
        patterns = [
            r'href="([^"]*\.pdf[^"]*)"',
            r'href="([^"]*bitstream[^"]*\.pdf[^"]*)"',
            r'href="([^"]*download[^"]*\.pdf[^"]*)"',
            r'href="([^"]*\.pdf)"',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                if match.startswith("http"):
                    return match
                elif match.startswith("/"):
                    # Relative URL
                    parsed = urlparse(base_url)
                    return f"{parsed.scheme}://{parsed.netloc}{match}"
                else:
                    # Relative to current path
                    base_path = base_url.rstrip("/")
                    return f"{base_path}/{match}"

        return None
