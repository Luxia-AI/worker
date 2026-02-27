import pytest

from app.services.verdict.verdict_generator import VerdictGenerator


class _TrustedSearchStub:
    async def generate_search_queries(self, **kwargs):  # noqa: ANN003
        return ["vitamin c immune health evidence", "vitamin c does not support immune health"]

    async def search(self, query, max_results=5):  # noqa: ANN001, ANN002
        return [
            {"url": "https://pubmed.ncbi.nlm.nih.gov/12345/"},
            {"url": "https://pubmed.ncbi.nlm.nih.gov/12345/#abstract"},
        ]


class _ScraperStub:
    async def scrape_one(self, session, url):  # noqa: ANN001, ANN002
        return {"url": url, "content": "Vitamin C supports immune health.", "source": "pubmed"}


class _FactExtractorStub:
    async def extract(self, pages, **kwargs):  # noqa: ANN001, ANN003
        return [
            {
                "statement": "Vitamin C supports immune health.",
                "confidence": 0.9,
                "source_url": pages[0].get("url", ""),
            }
        ]


@pytest.mark.asyncio
async def test_web_boost_deduplicates_attempted_urls_across_rounds():
    vg = VerdictGenerator.__new__(VerdictGenerator)
    vg.trusted_search = _TrustedSearchStub()
    vg.scraper = _ScraperStub()
    vg.fact_extractor = _FactExtractorStub()
    vg._last_predicate_queries_generated = []

    attempted = set()
    first = await vg._fetch_web_evidence_for_unknown_segments(
        ["Vitamin C does not support immune health"],
        max_queries_per_segment=2,
        max_urls_per_query=2,
        attempted_urls=attempted,
    )
    second = await vg._fetch_web_evidence_for_unknown_segments(
        ["Vitamin C does not support immune health"],
        max_queries_per_segment=2,
        max_urls_per_query=2,
        attempted_urls=attempted,
    )

    assert len(first) == 1
    assert len(second) == 0
