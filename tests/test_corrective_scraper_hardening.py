import pytest

from app.services.corrective.scraper import Scraper


@pytest.mark.asyncio
async def test_scraper_blocks_domain_after_persistent_403(monkeypatch):
    scraper = Scraper()

    async def fake_fetch_html(session, url):  # noqa: ANN001
        return None, True, 403

    async def fake_playwright(url, timeout_override=None):  # noqa: ANN001
        return None

    monkeypatch.setattr(scraper, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(scraper, "playwright_fallback", fake_playwright)

    result = await scraper.scrape_one(session=None, url="https://who.int/test-page")
    assert result["content"] is None
    assert scraper._should_skip_domain("https://who.int/another-page") is True


@pytest.mark.asyncio
async def test_scraper_attempted_urls_prevent_rescrape(monkeypatch):
    scraper = Scraper()
    scraper.reset_job_attempts()

    async def fake_fetch_html(session, url):  # noqa: ANN001
        return "<html><body>hello</body></html>", False, 200

    monkeypatch.setattr(scraper, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(scraper, "extract_text", lambda html: "hello")

    first = await scraper.scrape_one(session=None, url="https://cdc.gov/article")
    second = await scraper.scrape_one(session=None, url="https://cdc.gov/article")

    assert first["content"] == "hello"
    assert second["content"] is None
