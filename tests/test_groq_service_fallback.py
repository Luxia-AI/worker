from types import SimpleNamespace

import pytest

from app.services.llms import groq_service as groq_module


class _FakeCompletions:
    def __init__(self, api_key: str | None, state: dict) -> None:
        self.api_key = api_key
        self.state = state

    async def create(self, **kwargs):  # noqa: ANN003
        self.state["calls"].append(self.api_key)
        if self.api_key == "primary-key":
            raise RuntimeError("primary client failed")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])


class _FakeAsyncGroq:
    def __init__(self, api_key=None, default_headers=None, **kwargs):  # noqa: ANN001, ANN003
        self.api_key = api_key
        self.default_headers = default_headers
        state = kwargs.pop("_state")
        state["constructed"].append((api_key, default_headers))
        self.chat = SimpleNamespace(completions=_FakeCompletions(api_key, state))


@pytest.mark.asyncio
async def test_groq_service_falls_back_to_secondary_credentials(monkeypatch):
    state = {"constructed": [], "calls": []}

    def _factory(**kwargs):
        kwargs["_state"] = state
        return _FakeAsyncGroq(**kwargs)

    monkeypatch.setattr(groq_module, "AsyncGroq", _factory)
    monkeypatch.setenv("GROQ_API_KEY", "primary-key")
    monkeypatch.setenv("GROQ_API_KEY_FALLBACK", "fallback-key")
    monkeypatch.setenv("GROQ_ORG_ID_FALLBACK", "org-fallback")
    monkeypatch.setenv("GROQ_PROJECT_ID_FALLBACK", "project-fallback")

    svc = groq_module.GroqService()
    result = await svc.ainvoke("test", response_format="json", max_retries=1)

    assert result == {"ok": True}
    assert state["calls"] == ["primary-key", "fallback-key"]
    fallback_headers = next(headers for key, headers in state["constructed"] if key == "fallback-key")
    assert fallback_headers["x-groq-organization"] == "org-fallback"
    assert fallback_headers["x-groq-project"] == "project-fallback"


@pytest.mark.asyncio
async def test_groq_service_works_with_fallback_key_only(monkeypatch):
    state = {"constructed": [], "calls": []}

    def _factory(**kwargs):
        kwargs["_state"] = state
        return _FakeAsyncGroq(**kwargs)

    monkeypatch.setattr(groq_module, "AsyncGroq", _factory)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY_FALLBACK", "fallback-key")

    svc = groq_module.GroqService()
    result = await svc.ainvoke("test", response_format="json", max_retries=1)

    assert result == {"ok": True}
    assert state["calls"] == ["fallback-key"]
