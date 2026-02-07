import pytest

from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy
from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    # Avoid __init__ because it requires external services/keys.
    return VerdictGenerator.__new__(VerdictGenerator)


def test_split_claim_into_segments_expands_list_claim_meaningfully():
    vg = _vg()
    claim = (
        "A diet rich in fruits, vegetables, and low in saturated fats helps prevent "
        "noncommunicable diseases like diabetes and cancer."
    )

    segments = vg._split_claim_into_segments(claim)

    assert len(segments) >= 3
    assert not any(seg.strip().lower() == "vegetables" for seg in segments)
    assert all("helps prevent" in seg.lower() for seg in segments)
    assert not any("a diet a diet" in seg.lower() for seg in segments)


def test_should_rebuild_claim_breakdown_for_fragmentary_segments():
    vg = _vg()
    claim_breakdown = [
        {"claim_segment": "vegetables", "status": "UNKNOWN"},
        {"claim_segment": "low in saturated fats", "status": "PARTIALLY_VALID"},
    ]
    assert vg._should_rebuild_claim_breakdown(claim_breakdown) is True


@pytest.mark.asyncio
async def test_parse_verdict_result_rebuilds_low_quality_breakdown():
    vg = _vg()

    claim = (
        "A diet rich in fruits, vegetables, and low in saturated fats helps prevent "
        "noncommunicable diseases like diabetes and cancer."
    )
    evidence = [
        {
            "statement": "A diet rich in fruits helps prevent noncommunicable diseases.",
            "source_url": "https://nih.gov/example1",
            "final_score": 0.75,
            "credibility": 0.95,
        },
        {
            "statement": "A diet low in saturated fats helps prevent diabetes and cancer risk factors.",
            "source_url": "https://cdc.gov/example2",
            "final_score": 0.72,
            "credibility": 0.9,
        },
    ]

    llm_result = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.4,
        "rationale": "test",
        "claim_breakdown": [
            {"claim_segment": "vegetables", "status": "UNKNOWN", "supporting_fact": "", "source_url": ""}
        ],
    }

    out = vg._parse_verdict_result(llm_result, claim, evidence)
    assert out["claim_breakdown"]
    assert not any(item.get("claim_segment", "").strip().lower() == "vegetables" for item in out["claim_breakdown"])


def test_adaptive_decompose_claim_avoids_rich_in_low_in_phrase():
    policy = AdaptiveTrustPolicy()
    claim = (
        "A diet rich in fruits, vegetables, and low in saturated fats helps prevent "
        "noncommunicable diseases like diabetes and cancer."
    )
    parts = policy.decompose_claim(claim)
    assert not any("rich in low in" in p.lower() for p in parts)


def test_confidence_high_with_strong_supported_evidence():
    vg = _vg()
    evidence = [
        {"final_score": 0.82, "credibility": 0.95},
        {"final_score": 0.79, "credibility": 0.9},
        {"final_score": 0.77, "credibility": 0.92},
        {"final_score": 0.81, "credibility": 0.93},
        {"final_score": 0.76, "credibility": 0.9},
    ]
    claim_breakdown = [
        {"status": "VALID"},
        {"status": "PARTIALLY_VALID"},
        {"status": "VALID"},
    ]
    conf = vg._calculate_confidence(evidence, claim_breakdown)
    assert conf >= 0.75


def test_confidence_high_with_strong_contradicting_evidence():
    vg = _vg()
    evidence = [
        {"final_score": 0.84, "credibility": 0.95, "source_url": "https://nih.gov/a"},
        {"final_score": 0.80, "credibility": 0.92, "source_url": "https://cdc.gov/b"},
        {"final_score": 0.78, "credibility": 0.90, "source_url": "https://who.int/c"},
    ]
    claim_breakdown = [
        {"status": "INVALID"},
        {"status": "PARTIALLY_INVALID"},
    ]
    conf = vg._calculate_confidence(evidence, claim_breakdown)
    assert conf >= 0.75


def test_parse_verdict_result_normalizes_duplicated_segment_prefix():
    vg = _vg()
    claim = "A diet rich in fruits, vegetables, and low in saturated fats helps prevent NCDs."
    evidence = [
        {
            "statement": "A diet rich in vegetables helps reduce noncommunicable disease risk.",
            "source_url": "https://nih.gov/example",
            "final_score": 0.8,
            "credibility": 0.9,
        }
    ]
    llm_result = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.5,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "A diet A diet rich in vegetables helps prevent NCDs",
                "status": "PARTIALLY_VALID",
                "supporting_fact": "A diet rich in vegetables helps reduce noncommunicable disease risk.",
                "source_url": "https://nih.gov/example",
            }
        ],
    }

    out = vg._parse_verdict_result(llm_result, claim, evidence)
    segments = [b.get("claim_segment", "").lower() for b in out["claim_breakdown"]]
    assert all("a diet a diet" not in s for s in segments)


def test_parse_verdict_result_flips_valid_when_supporting_fact_negates_claim():
    vg = _vg()
    claim = "Drinking at least eight glasses of water a day is essential."
    evidence = [
        {
            "statement": "Drinking 8 glasses of water a day is not necessary for everyone.",
            "source_url": "https://example.org/water",
            "final_score": 0.8,
            "credibility": 0.9,
        }
    ]
    llm_result = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Drinking at least eight glasses of water a day is essential",
                "status": "VALID",
                "supporting_fact": "Drinking 8 glasses of water a day is not necessary for everyone.",
                "source_url": "https://example.org/water",
            }
        ],
    }

    out = vg._parse_verdict_result(llm_result, claim, evidence)
    assert out["claim_breakdown"][0]["status"] in {"INVALID", "PARTIALLY_INVALID"}
