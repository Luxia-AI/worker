import pytest

from app.services.llms import hybrid_service as hybrid_module


class _FakeGroqService:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        return {"text": "ok"}


@pytest.mark.asyncio
async def test_groq_never_exceeds_per_job_limit(monkeypatch):
    monkeypatch.setenv("ALLOW_GROQ_BURST", "false")
    monkeypatch.setenv("GROQ_MAX_CALLS_PER_JOB", "2")
    monkeypatch.setenv("LOCAL_LLM_ENABLED", "false")

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
    monkeypatch.setenv("LOCAL_LLM_ENABLED", "false")

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
