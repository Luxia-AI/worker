import pytest

from app.services.corrective.trusted_search import TrustedSearch


@pytest.mark.asyncio
async def test_reformulate_queries_retries_on_json_failure_then_succeeds():
    ts = TrustedSearch.__new__(TrustedSearch)

    class _LLMStub:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError(
                    "json_validate_failed: max completion tokens reached before generating a valid document"
                )
            return {"queries": ["vitamin d cancer evidence", "vitamin d oncology study", "extra 1", "extra 2"]}

    ts.llm_client = _LLMStub()
    queries = await ts.reformulate_queries("Vitamin D and cancer", ["vitamin d", "cancer"])
    assert queries
    assert isinstance(queries, list)
    assert ts.llm_client.calls >= 2
