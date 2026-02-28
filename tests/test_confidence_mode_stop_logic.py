from app.services.corrective.pipeline import CorrectivePipeline


def test_confidence_mode_does_not_stop_on_coverage_only_for_strong_therapeutic_claim():
    claim_frame = {"is_strong_therapeutic": True}
    adaptive = {
        "coverage": 0.70,
        "diversity": 0.0,
        "strong_covered": 0,
        "is_sufficient": False,
        "contradicted_subclaims": 0,
    }
    should_stop, reason = CorrectivePipeline._should_stop_confidence_mode(
        claim_frame=claim_frame,
        adaptive_trust=adaptive,
        confidence_target_coverage=0.50,
        confidence_max_new_trusted_urls=12,
        new_trusted_urls_processed=3,
    )
    assert should_stop is False
    assert "continue" in reason


def test_confidence_mode_does_not_stop_when_coverage_reached_but_not_sufficient():
    claim_frame = {"is_strong_therapeutic": False}
    adaptive = {
        "coverage": 0.72,
        "diversity": 0.62,
        "strong_covered": 1,
        "is_sufficient": False,
        "contradicted_subclaims": 0,
        "trust_post": 0.58,
        "agreement": 0.9,
        "num_subclaims": 1,
    }
    should_stop, reason = CorrectivePipeline._should_stop_confidence_mode(
        claim_frame=claim_frame,
        adaptive_trust=adaptive,
        confidence_target_coverage=0.50,
        confidence_max_new_trusted_urls=12,
        new_trusted_urls_processed=4,
    )
    assert should_stop is False
    assert "adaptive_sufficient=False" in reason


def test_confidence_mode_stops_on_low_yield_saturation():
    claim_frame = {"is_strong_therapeutic": False}
    adaptive = {
        "coverage": 0.95,
        "diversity": 0.95,
        "strong_covered": 0,
        "is_sufficient": False,
        "contradicted_subclaims": 0,
        "trust_post": 0.99,
        "agreement": 0.95,
        "num_subclaims": 1,
    }
    should_stop, reason = CorrectivePipeline._should_stop_confidence_mode(
        claim_frame=claim_frame,
        adaptive_trust=adaptive,
        confidence_target_coverage=0.50,
        confidence_max_new_trusted_urls=12,
        new_trusted_urls_processed=8,
        zero_extraction_rounds=3,
        low_yield_rounds=3,
    )
    assert should_stop is True
    assert "low-yield" in reason or "expected-gain" in reason
