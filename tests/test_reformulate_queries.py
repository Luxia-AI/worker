import pytest

from app.services.corrective.trusted_search import TrustedSearch


@pytest.mark.asyncio
async def test_reformulate_queries():
    ts = TrustedSearch()
    queries = await ts.reformulate_queries("Vitamin D cures cancer", ["vitamin d", "cancer"])
    print(queries)
