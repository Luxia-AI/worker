import pytest

from app.services.corrective.scraper import Scraper


@pytest.mark.asyncio
async def test_scraper():
    s = Scraper()
    urls = ["https://www.cdc.gov/coronavirus/2019-ncov/vaccines/effectiveness.html"]
    results = await s.scrape_all(urls)
    print(results)
