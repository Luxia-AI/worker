import pytest

from app.services.corrective.scraper import Scraper


@pytest.mark.asyncio
async def test_scraper():
    s = Scraper()
    urls = ["https://cdc.gov/healthy-weight-growth/about/tips-for-balancing-food-activity.html"]
    results = await s.scrape_all(urls)
    print(results)
