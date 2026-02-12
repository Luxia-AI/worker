import pytest

from app.constants.config import LLM_MAX_TOKENS_QUERY_REFORMULATION, LLM_MAX_TOKENS_VERDICT_GENERATION
from app.services.llms import hybrid_service as hybrid_module


class _FakeGroqService:
    def __init__(self):
        self.calls = 0
        self.max_tokens_seen = []

    async def ainvoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        self.max_tokens_seen.append(kwargs.get("max_tokens"))
        return {"text": "ok"}


@pytest.mark.asyncio
async def test_groq_never_exceeds_per_job_limit(monkeypatch):
    monkeypatch.setenv("ALLOW_GROQ_BURST", "false")
    monkeypatch.setenv("GROQ_MAX_CALLS_PER_JOB", "2")
    monkeypatch.setenv("GROQ_RESERVED_VERDICT_CALLS", "0")
    monkeypatch.setenv("GROQ_RESERVED_FACT_EXTRACTION_CALLS", "0")

    fake = _FakeGroqService()
    monkeypatch.setattr(hybrid_module, "GroqService", lambda: fake)

    svc = hybrid_module.HybridLLMService()
    hybrid_module.reset_groq_counter(job_id="job-1", max_calls=2)

    for _ in range(5):
        result = await svc.ainvoke(
            "non-critical",
            response_format="json",
            priority=hybrid_module.LLMPriority.LOW,
            call_tag="relation_extraction",
        )
        assert isinstance(result, dict)

    assert fake.calls == 2
    meta = hybrid_module.get_groq_job_metadata()
    assert meta["groq_calls_used"] == 2
    assert meta["degraded_mode"] is True


@pytest.mark.asyncio
async def test_groq_limit_is_per_job(monkeypatch):
    monkeypatch.setenv("ALLOW_GROQ_BURST", "false")

    fake = _FakeGroqService()
    monkeypatch.setattr(hybrid_module, "GroqService", lambda: fake)

    svc = hybrid_module.HybridLLMService()

    hybrid_module.reset_groq_counter(job_id="job-a", max_calls=1)
    await svc.ainvoke("critical", priority=hybrid_module.LLMPriority.HIGH, call_tag="verdict_generation")
    with pytest.raises(RuntimeError):
        await svc.ainvoke("critical", priority=hybrid_module.LLMPriority.HIGH, call_tag="verdict_generation")

    hybrid_module.reset_groq_counter(job_id="job-b", max_calls=1)
    await svc.ainvoke("critical", priority=hybrid_module.LLMPriority.HIGH, call_tag="verdict_generation")

    assert fake.calls == 2


@pytest.mark.asyncio
async def test_reserve_one_verdict_call_budget(monkeypatch):
    monkeypatch.setenv("ALLOW_GROQ_BURST", "false")
    monkeypatch.setenv("GROQ_MAX_CALLS_PER_JOB", "2")
    monkeypatch.setenv("GROQ_RESERVED_VERDICT_CALLS", "1")

    fake = _FakeGroqService()
    monkeypatch.setattr(hybrid_module, "GroqService", lambda: fake)

    svc = hybrid_module.HybridLLMService()
    hybrid_module.reset_groq_counter(job_id="job-reserve", max_calls=2)

    await svc.ainvoke("q1", priority=hybrid_module.LLMPriority.HIGH, call_tag="query_reformulation")
    with pytest.raises(RuntimeError, match="reserved for verdict"):
        await svc.ainvoke("q2", priority=hybrid_module.LLMPriority.HIGH, call_tag="query_reformulation")
    await svc.ainvoke("verdict", priority=hybrid_module.LLMPriority.HIGH, call_tag="verdict_generation")

    assert fake.calls == 2


@pytest.mark.asyncio
async def test_call_tag_specific_max_tokens_are_forwarded(monkeypatch):
    monkeypatch.setenv("ALLOW_GROQ_BURST", "false")
    monkeypatch.setenv("GROQ_MAX_CALLS_PER_JOB", "3")
    monkeypatch.setenv("GROQ_RESERVED_VERDICT_CALLS", "1")

    fake = _FakeGroqService()
    monkeypatch.setattr(hybrid_module, "GroqService", lambda: fake)

    svc = hybrid_module.HybridLLMService()
    hybrid_module.reset_groq_counter(job_id="job-tokens", max_calls=3)

    await svc.ainvoke(
        "reformulate",
        response_format="json",
        priority=hybrid_module.LLMPriority.HIGH,
        call_tag="query_reformulation",
    )
    await svc.ainvoke(
        "verdict",
        response_format="json",
        priority=hybrid_module.LLMPriority.HIGH,
        call_tag="verdict_generation",
    )

    assert fake.max_tokens_seen[0] == LLM_MAX_TOKENS_QUERY_REFORMULATION
    assert fake.max_tokens_seen[1] == LLM_MAX_TOKENS_VERDICT_GENERATION


@pytest.mark.asyncio
async def test_fact_extraction_can_use_reserved_critical_pool(monkeypatch):
    monkeypatch.setenv("ALLOW_GROQ_BURST", "false")
    monkeypatch.setenv("GROQ_MAX_CALLS_PER_JOB", "4")
    monkeypatch.setenv("GROQ_RESERVED_VERDICT_CALLS", "1")
    monkeypatch.setenv("GROQ_RESERVED_FACT_EXTRACTION_CALLS", "1")

    fake = _FakeGroqService()
    monkeypatch.setattr(hybrid_module, "GroqService", lambda: fake)

    svc = hybrid_module.HybridLLMService()
    hybrid_module.reset_groq_counter(job_id="job-critical", max_calls=4)

    await svc.ainvoke("q1", priority=hybrid_module.LLMPriority.HIGH, call_tag="query_reformulation")
    await svc.ainvoke("q2", priority=hybrid_module.LLMPriority.HIGH, call_tag="query_reformulation")
    # Low-priority non-critical extraction should be blocked once reserve pool is reached.
    skipped = await svc.ainvoke("rels", priority=hybrid_module.LLMPriority.LOW, call_tag="relation_extraction")
    assert skipped.get("_degraded_skip") is True

    # Fact extraction remains allowed from reserved critical budget.
    result = await svc.ainvoke(
        "facts",
        response_format="json",
        priority=hybrid_module.LLMPriority.LOW,
        call_tag="fact_extraction",
    )
    assert isinstance(result, dict)
    await svc.ainvoke("verdict", priority=hybrid_module.LLMPriority.HIGH, call_tag="verdict_generation")

    meta = hybrid_module.get_groq_job_metadata()
    assert meta["reserved_fact_extraction_calls"] == 1
    assert fake.calls == 4


@pytest.mark.asyncio
async def test_high_priority_entity_extraction_can_consume_non_verdict_reserved_slot(monkeypatch):
    monkeypatch.setenv("ALLOW_GROQ_BURST", "false")
    monkeypatch.setenv("GROQ_MAX_CALLS_PER_JOB", "4")
    monkeypatch.setenv("GROQ_RESERVED_VERDICT_CALLS", "1")
    monkeypatch.setenv("GROQ_RESERVED_FACT_EXTRACTION_CALLS", "1")

    fake = _FakeGroqService()
    monkeypatch.setattr(hybrid_module, "GroqService", lambda: fake)

    svc = hybrid_module.HybridLLMService()
    hybrid_module.reset_groq_counter(job_id="job-entity-critical", max_calls=4)

    await svc.ainvoke("q1", priority=hybrid_module.LLMPriority.HIGH, call_tag="query_reformulation")
    await svc.ainvoke("q2", priority=hybrid_module.LLMPriority.HIGH, call_tag="query_reformulation")
    # Remaining slots now equal reserve_pool (2). High-priority entity extraction should still run.
    await svc.ainvoke("entities", priority=hybrid_module.LLMPriority.HIGH, call_tag="entity_extraction")
    await svc.ainvoke("verdict", priority=hybrid_module.LLMPriority.HIGH, call_tag="verdict_generation")

    assert fake.calls == 4
