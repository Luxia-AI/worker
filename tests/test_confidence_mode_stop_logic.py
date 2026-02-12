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
