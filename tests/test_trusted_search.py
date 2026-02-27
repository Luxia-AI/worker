import pytest

from app.services.corrective import trusted_search as trusted_search_module


class _FakeLLM:
    async def ainvoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return {"queries": []}


@pytest.fixture
def trusted_search(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setattr(trusted_search_module, "HybridLLMService", _FakeLLM)
    return trusted_search_module.TrustedSearch()


def test_is_trusted_allowlist_and_override(trusted_search):
    assert trusted_search.is_trusted("https://pubmed.ncbi.nlm.nih.gov/123456/")
    assert not trusted_search.is_trusted("https://example.org/health")
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


def test_sanitize_query_fixes_negation_grammar(trusted_search):
    cleaned = trusted_search._sanitize_query("vitamin c do not support immune health")
    assert cleaned == "vitamin c does not support immune health"


def test_claim_logic_tracks_include_support_and_refute(trusted_search):
    tracks = trusted_search._build_claim_logic_query_tracks(
        "vitamin c",
        "support",
        "immune health",
        "Vitamin C does not support immune health",
    )
    joined = " | ".join(tracks).lower()
    assert "vitamin c support immune health" in joined
    assert "does not" in joined or "no evidence" in joined


@pytest.mark.asyncio
async def test_google_query_cleanup_strips_confidence_negatives(monkeypatch, trusted_search):
    trusted_search.google_available = True
    trusted_search.serper_available = False
    trusted_search.google_quota_exceeded = False
    trusted_search.min_allowlist_pass = 1

    captured = {}

    async def _fake_google_search(session, query):
        captured["query"] = query
        return ([{"link": "https://pubmed.ncbi.nlm.nih.gov/123456/"}], False)

    monkeypatch.setattr(trusted_search, "search_query_google", _fake_google_search)

    urls = await trusted_search.search_query(
        session=None,
        query='site:nih.gov "lactose digestion" -facebook -quora',
    )
    assert urls == ["https://pubmed.ncbi.nlm.nih.gov/123456/"]
    assert "-facebook" not in captured["query"].lower()
    assert "-quora" not in captured["query"].lower()
