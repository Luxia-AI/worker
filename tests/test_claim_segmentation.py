from app.services.common.claim_segmentation import split_claim_into_segments
from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy
from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    return VerdictGenerator.__new__(VerdictGenerator)


def test_or_disjunction_expansion_preserves_predicate():
    claim = "Vaccines do not cause autism or the flu."
    segments = split_claim_into_segments(claim)

    assert len(segments) >= 2
    assert any("do not cause autism" in s.lower() for s in segments)
    assert any("do not cause the flu" in s.lower() for s in segments)


def test_no_fragment_segments():
    claim = "Vaccines do not cause autism or the flu."
    segments = split_claim_into_segments(claim)

    bad = {"or the flu", "the flu"}
    assert not any(s.strip().lower() in bad for s in segments)


def test_adaptive_and_verdict_use_identical_segments():
    claim = "Vaccines do not cause autism or the flu."
    adaptive = AdaptiveTrustPolicy().decompose_claim(claim)
    verdict_segments = _vg()._split_claim_into_segments(claim)
    assert adaptive == verdict_segments
