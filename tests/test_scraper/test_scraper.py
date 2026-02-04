"""
Unit tests for WHO/IRIS/IMSEAR scraper with recorded fixtures.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Response

from app.services.scraper import URLType, WHOScraper


class TestURLClassification:
    """Test URL classification."""

    def test_pdf_direct(self):
        scraper = WHOScraper()
        assert scraper.classify_url("https://www.who.int/document.pdf") == URLType.PDF_DIRECT
        assert scraper.classify_url("https://iris.who.int/bitstream/handle/123/456/document.pdf") == URLType.PDF_DIRECT

    def test_who_api_bitstream(self):
        scraper = WHOScraper()
        assert (
            scraper.classify_url("https://www.who.int/server/api/core/bitstreams/123-456-789/content")
            == URLType.WHO_API_BITSTREAM
        )

    def test_imsear_download(self):
        scraper = WHOScraper()
        assert scraper.classify_url("https://imsear.searo.who.int/handle/123/456/download") == URLType.IMSEAR_DOWNLOAD

    def test_iris_html(self):
        scraper = WHOScraper()
        assert scraper.classify_url("https://iris.who.int/handle/123/456") == URLType.IRIS_HTML

    def test_generic_html(self):
        scraper = WHOScraper()
        assert scraper.classify_url("https://example.com/page") == URLType.GENERIC_HTML
        assert scraper.classify_url("https://www.who.int/page") == URLType.GENERIC_HTML


class TestScraper:
    """Test scraper functionality."""

    @pytest.fixture
    async def scraper(self):
        """Create scraper instance."""
        async with WHOScraper() as s:
            yield s

    @pytest.fixture
    def mock_pdf_response(self):
        """Mock PDF response."""
        # Create a minimal PDF content (this is a real minimal PDF)
        pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n"
            b"3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n"
            b"4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Hello World) Tj\nET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n0000000200 00000 n \n"
            b"trailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n284\n%%EOF"
        )

        response = MagicMock(spec=Response)
        response.status_code = 200
        response.content = pdf_content
        response.headers = {"content-type": "application/pdf"}
        response.text = ""  # Not used for PDFs
        return response

    @pytest.fixture
    def mock_html_response(self):
        """Mock HTML response."""
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
        <h1>Test Document</h1>
        <p>This is a test document with some content.</p>
        <div class="content">
        <p>More content here.</p>
        </div>
        </body>
        </html>
        """
        response = MagicMock(spec=Response)
        response.status_code = 200
        response.content = html_content.encode()
        response.headers = {"content-type": "text/html"}
        response.text = html_content
        return response

    @pytest.fixture
    def mock_iris_landing_page(self):
        """Mock IRIS landing page with PDF link."""
        html_content = """
        <html>
        <body>
        <h1>Document Title</h1>
        <a href="/bitstream/handle/123/456/document.pdf">Download PDF</a>
        <a href="https://iris.who.int/bitstream/handle/123/456/document.pdf">Direct PDF</a>
        </body>
        </html>
        """
        response = MagicMock(spec=Response)
        response.status_code = 200
        response.content = html_content.encode()
        response.headers = {"content-type": "text/html"}
        response.text = html_content
        return response

    @patch("httpx.AsyncClient")
    async def test_scrape_pdf_direct_success(self, mock_client_class, scraper, mock_pdf_response):
        """Test successful PDF scraping."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_pdf_response

        result = await scraper.scrape("https://www.who.int/document.pdf")

        assert result.status == "success"
        assert result.content_type == "pdf"
        assert result.method_used == "http"
        assert "Hello World" in result.content
        assert result.chars_extracted > 0

    @patch("httpx.AsyncClient")
    async def test_scrape_html_success(self, mock_client_class, scraper, mock_html_response):
        """Test successful HTML scraping."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_html_response

        result = await scraper.scrape("https://www.who.int/page")

        assert result.status == "success"
        assert result.content_type == "html"
        assert result.method_used == "http"
        assert "Test Document" in result.content
        assert result.chars_extracted > 0

    @patch("httpx.AsyncClient")
    async def test_scrape_iris_landing_page_with_pdf_link(
        self, mock_client_class, scraper, mock_iris_landing_page, mock_pdf_response
    ):
        """Test IRIS landing page that redirects to PDF."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # First call returns landing page, second returns PDF
        mock_client.get.side_effect = [mock_iris_landing_page, mock_pdf_response]

        result = await scraper.scrape("https://iris.who.int/handle/123/456")

        assert result.status == "success"
        assert result.content_type == "pdf"
        assert "Hello World" in result.content

    @patch("httpx.AsyncClient")
    async def test_scrape_http_403_fallback(self, mock_client_class, scraper, mock_pdf_response):
        """Test HTTP 403 fallback."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # First call fails with 403, second succeeds
        error_response = MagicMock(spec=Response)
        error_response.status_code = 403

        mock_client.get.side_effect = [error_response, mock_pdf_response]
        mock_client.head.return_value = mock_pdf_response

        result = await scraper.scrape("https://www.who.int/document.pdf")

        assert result.status == "success"
        assert result.content_type == "pdf"
        assert result.method_used == "http_fallback"

    async def test_scrape_playwright_fallback(self, scraper):
        """Test Playwright fallback (mocked)."""
        # This would require more complex mocking of Playwright
        # For now, just test that the method exists
        assert hasattr(scraper, "_scrape_playwright")


class TestMetrics:
    """Test scraper metrics."""

    def test_metrics_recording(self):
        """Test metrics recording."""
        metrics = WHOScraper().metrics

        # Record some attempts
        metrics.record_attempt("who.int", "http", True, 1.0)
        metrics.record_attempt("who.int", "http", False, 2.0, "403")
        metrics.record_attempt("iris.who.int", "playwright", True, 1.5)

        # Check success rates
        rates = metrics.get_success_rate_by_domain()
        assert rates["who.int"] == 0.5  # 1/2
        assert rates["iris.who.int"] == 1.0  # 1/1

        # Check latencies
        latencies = metrics.get_avg_latency_by_method()
        assert latencies["http"] == 1.5  # (1.0 + 2.0) / 2
        assert latencies["playwright"] == 1.5

        # Check errors
        errors = metrics.get_error_counts()
        assert errors["403"] == 1


class TestFixtures:
    """Test fixtures for different scenarios."""

    def test_bitstream_fixture(self):
        """Test WHO API bitstream URL."""
        scraper = WHOScraper()
        url = "https://www.who.int/server/api/core/bitstreams/123e4567-e89b-12d3-a456-426614174000/content"
        assert scraper.classify_url(url) == URLType.WHO_API_BITSTREAM

    def test_imsear_download_fixture(self):
        """Test IMSEAR download URL."""
        scraper = WHOScraper()
        url = "https://imsear.searo.who.int/handle/123/456/download"
        assert scraper.classify_url(url) == URLType.IMSEAR_DOWNLOAD

    def test_iris_html_fixture(self):
        """Test IRIS HTML landing page."""
        scraper = WHOScraper()
        url = "https://iris.who.int/handle/123/456"
        assert scraper.classify_url(url) == URLType.IRIS_HTML
