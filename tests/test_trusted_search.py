import pytest

from app.services.corrective.trusted_search import TrustedSearch


@pytest.mark.asyncio
async def test_trusted_search():
    ts = TrustedSearch()

    post = "Vitamin D cures cancer"
    failed_entities = ["vitamin D", "cancer"]

    urls = await ts.run(post_text=post, failed_entities=failed_entities)

    print("\n===== TRUSTED SEARCH RESULTS =====")
    for u in urls:
        print(u)

    assert urls is not None
    assert isinstance(urls, list)
