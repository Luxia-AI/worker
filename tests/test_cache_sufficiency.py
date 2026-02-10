from app.services.corrective.pipeline import CorrectivePipeline


def test_cache_does_not_short_circuit_with_unresolved_subclaim():
    adaptive_trust = {"is_sufficient": True, "trust_post": 0.82}
    verdict_result = {"required_segments_resolved": False}
    assert not CorrectivePipeline._cache_fast_path_allowed(adaptive_trust, verdict_result)


def test_cache_short_circuit_when_all_segments_resolved():
    adaptive_trust = {"is_sufficient": True, "trust_post": 0.82}
    verdict_result = {"required_segments_resolved": True}
    assert CorrectivePipeline._cache_fast_path_allowed(adaptive_trust, verdict_result)


def test_threshold_met_not_hardcoded_true_in_cache_response():
    payload = CorrectivePipeline._trust_payload_adaptive({"is_sufficient": False, "trust_post": 0.88})
    assert payload["trust_threshold_met"] is False
