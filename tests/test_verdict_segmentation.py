import pytest

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
