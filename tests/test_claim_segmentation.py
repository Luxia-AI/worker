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


def test_contrast_not_clause_is_split_into_two_verifiable_segments():
    claim = "They kill bacteria, not the viruses that cause cold and flu."
    segments = split_claim_into_segments(claim)

    assert len(segments) == 2
    joined = " | ".join(segments).lower()
    assert "kill bacteria" in joined
    assert "virus" in joined
    assert "not" in joined


def test_segments_require_predicates_and_merge_fragments():
    claim = "A healthy diet rich in vegetables and fruit may help reduce the risk of some cancers."
    segments = split_claim_into_segments(claim)

    assert len(segments) == 1
    segment = segments[0].lower()
    assert "may help reduce" in segment
    assert not segment.startswith("and ")


def test_parenthetical_and_does_not_split_into_fragment_segments():
    claim = (
        "Low-fat diets rich in fruits and vegetables (foods that are low in fat and may contain dietary fiber "
        "and Vitamin A or Vitamin C) may reduce the risk of some types of cancer."
    )
    segments = split_claim_into_segments(claim)

    # Should not produce fragmentary parts from conjunctions inside parentheses.
    assert all("(" not in s or ")" in s for s in segments)
    assert not any(s.strip().lower().startswith("may contain dietary fiber") for s in segments)
    assert all(any(v in s.lower() for v in ["may", "reduce", "contain", "helps", "is", "are"]) for s in segments)


def test_semicolon_claim_does_not_cross_merge_subjects():
    claim = (
        "Most of your body's tissues are constantly renewing; "
        "your skeleton replaces itself roughly every 10 years and your stomach lining every 3-4 days"
    )
    segments = split_claim_into_segments(claim)

    assert len(segments) == 2
    second = segments[1].lower()
    assert "your stomach lining" in second
    assert "most of your body's tissues are your stomach lining" not in second


def test_aux_clause_inherits_subject_after_and_split():
    claim = (
        "Laughter releases endorphins that can decrease pain and "
        "has been shown to help lower blood sugar levels after a meal"
    )
    segments = split_claim_into_segments(claim)

    assert len(segments) == 2
    assert segments[1].lower().startswith("laughter has been shown to help lower blood sugar")
