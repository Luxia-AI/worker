import logging

from app.core.logger import is_debug_enabled, log_value_payload, truncate_list_payload


def test_is_debug_enabled_toggle(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    assert is_debug_enabled() is True
    monkeypatch.setenv("DEBUG", "false")
    assert is_debug_enabled() is False


def test_truncate_list_payload_caps(monkeypatch):
    monkeypatch.setenv("LOG_VALUE_SAMPLE_LIMIT", "2")
    monkeypatch.setenv("LOG_VALUE_MAX_ITEMS", "3")
    payload = {"items": [1, 2, 3, 4], "nested": [{"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}]}
    out = truncate_list_payload(payload)
    assert out["items"] == [1, 2]
    assert len(out["nested"]) == 2


def test_log_value_payload_emits_phase_output(caplog):
    logger = logging.getLogger("test.value.payload")
    logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO, logger="test.value.payload")
    log_value_payload(logger, "ranking", {"top_score": 0.71, "queries_used": ["a", "b", "c"]})
    assert "[PhaseOutput]" in caplog.text
    assert '"phase": "ranking"' in caplog.text
    assert '"top_score": 0.71' in caplog.text
