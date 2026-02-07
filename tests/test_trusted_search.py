import pytest

from app.services.corrective import trusted_search as trusted_search_module


class _FakeLLM:
    async def ainvoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return {"queries": []}


@pytest.fixture
def trusted_search(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setenv("TRUSTED_SEARCH_EXTRA_ALLOWLIST", "example.org")
    monkeypatch.setattr(trusted_search_module, "HybridLLMService", _FakeLLM)
    return trusted_search_module.TrustedSearch()


def test_is_trusted_allowlist_and_override(trusted_search):
    assert trusted_search.is_trusted("https://pubmed.ncbi.nlm.nih.gov/123456/")
    assert trusted_search.is_trusted("https://example.org/health")
    assert not trusted_search.is_trusted("https://not-trusted.invalid/path")


@pytest.mark.parametrize(
    "url",
    [
        "data:text/plain;base64,SGVsbG8=",
        "https://pubmed.ncbi.nlm.nih.gov/?pano=1",
        "https://who.int/a?blob=QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=",
        f"https://who.int/path?{'x=' + 'a' * 500}",
    ],
)
def test_url_quality_filter_rejects_junk(url, trusted_search):
    assert trusted_search._url_quality_reject_reason(url) is not None


def test_filter_trusted_urls_reports_rejection_counts(trusted_search):
    items = [
        {"link": "https://not-trusted.invalid/path", "title": "x", "snippet": "x"},
        {"link": "data:text/plain;base64,SGVsbG8=", "title": "x", "snippet": "x"},
        {"link": "https://pubmed.ncbi.nlm.nih.gov/123456/", "title": "Study", "snippet": "Results"},
    ]
    urls, metrics = trusted_search._filter_trusted_urls(items, provider="Google", query="query")

    assert urls == ["https://pubmed.ncbi.nlm.nih.gov/123456/"]
    assert metrics["count_rejected_by_allowlist"] == 2
    assert metrics["count_rejected_by_quality"] == 0
    assert metrics["final_trusted_count"] == 1
