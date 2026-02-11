from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    return VerdictGenerator.__new__(VerdictGenerator)


def test_negative_segment_negative_evidence_entails_and_reconciles_true():
    vg = _vg()
    claim = "Antibiotics do not work against viruses."
    evidence = [
        {
            "statement": "Antibiotics do not treat viral infections.",
            "source_url": "https://www.cdc.gov/example",
            "final_score": 0.9,
            "credibility": 0.95,
        }
    ]
    llm_result = {
        "verdict": "FALSE",
        "confidence": 0.8,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Antibiotics do not work against viruses",
                "status": "VALID",
                "supporting_fact": "Antibiotics do not treat viral infections.",
                "source_url": "https://www.cdc.gov/example",
            }
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    polarity = vg._segment_polarity(
        "Antibiotics do not work against viruses",
        "Antibiotics do not treat viral infections.",
        stance="neutral",
    )
    parsed = vg._parse_verdict_result(llm_result, claim, evidence)

    assert polarity == "entails"
    assert parsed["verdict"] == "TRUE"


def test_positive_segment_negative_evidence_is_contradiction_and_false():
    vg = _vg()
    claim = "Antibiotics work against viruses."

    polarity = vg._segment_polarity(
        "Antibiotics work against viruses",
        "Antibiotics do not treat viral infections.",
        stance="neutral",
    )
    reconciled = vg._reconcile_verdict_with_breakdown(
        claim,
        [
            {
                "claim_segment": "Antibiotics work against viruses",
                "status": "INVALID",
                "supporting_fact": "Antibiotics do not treat viral infections.",
                "source_url": "https://www.cdc.gov/example",
            }
        ],
    )

    assert polarity == "contradicts"
    assert reconciled["verdict"] == "FALSE"


def test_ineffective_and_do_not_work_are_symmetric_entailment():
    vg = _vg()
    polarity = vg._segment_polarity(
        "Antibiotics are ineffective against viruses",
        "Antibiotics do not work against viruses.",
        stance="neutral",
    )
    assert polarity == "entails"
