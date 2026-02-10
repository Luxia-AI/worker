from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    return VerdictGenerator.__new__(VerdictGenerator)


def test_true_requires_all_segments_valid():
    vg = _vg()
    claim = "Vaccines do not cause autism or the flu."
    evidence = [
        {
            "statement": "Childhood vaccines do not cause autism.",
            "source_url": "https://who.int/autism",
            "final_score": 0.9,
            "credibility": 0.95,
        },
        {
            "statement": "Flu vaccines do not cause flu illness.",
            "source_url": "https://cdc.gov/flu-vaccine",
            "final_score": 0.88,
            "credibility": 0.95,
        },
    ]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vaccines do not cause autism",
                "status": "VALID",
                "supporting_fact": "Childhood vaccines do not cause autism.",
                "source_url": "https://who.int/autism",
            },
            {
                "claim_segment": "Vaccines do not cause the flu",
                "status": "VALID",
                "supporting_fact": "Flu vaccines do not cause flu illness.",
                "source_url": "https://cdc.gov/flu-vaccine",
            },
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)

    assert parsed["verdict"] == "TRUE"
    assert parsed["required_segments_resolved"] is True
    assert parsed["resolved_segments_count"] == parsed["required_segments_count"]


def test_unknown_segment_forces_non_decisive_verdict():
    vg = _vg()
    claim = "Vaccines do not cause autism or the flu."
    evidence = [
        {
            "statement": "Childhood vaccines do not cause autism.",
            "source_url": "https://who.int/autism",
            "final_score": 0.9,
            "credibility": 0.95,
        }
    ]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.95,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vaccines do not cause autism",
                "status": "VALID",
                "supporting_fact": "Childhood vaccines do not cause autism.",
                "source_url": "https://who.int/autism",
            }
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)

    assert parsed["verdict"] == "PARTIALLY_TRUE"
    assert parsed["required_segments_resolved"] is False
    assert parsed["unresolved_segments"] >= 1
    assert parsed["truthfulness_percent"] < 90.0


def test_partial_breakdown_forces_partially_true():
    vg = _vg()
    claim = "Vaccines do not cause autism or the flu."
    evidence = [
        {
            "statement": "Childhood vaccines do not cause autism.",
            "source_url": "https://who.int/autism",
            "final_score": 0.9,
            "credibility": 0.95,
        },
        {
            "statement": "Flu vaccination has not been shown to cause influenza infection.",
            "source_url": "https://cdc.gov/flu-vaccine",
            "final_score": 0.68,
            "credibility": 0.92,
        },
    ]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vaccines do not cause autism",
                "status": "VALID",
                "supporting_fact": "Childhood vaccines do not cause autism.",
                "source_url": "https://who.int/autism",
            },
            {
                "claim_segment": "Vaccines do not cause the flu",
                "status": "PARTIALLY_VALID",
                "supporting_fact": "Flu vaccination has not been shown to cause influenza infection.",
                "source_url": "https://cdc.gov/flu-vaccine",
            },
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)
    assert parsed["verdict"] == "PARTIALLY_TRUE"


def test_all_invalid_maps_to_false():
    vg = _vg()
    claim = "Vaccines cause autism or the flu."
    evidence = [
        {
            "statement": "Childhood vaccines do not cause autism.",
            "source_url": "https://who.int/autism",
            "final_score": 0.9,
            "credibility": 0.95,
        },
        {
            "statement": "Flu vaccines do not cause flu illness.",
            "source_url": "https://cdc.gov/flu-vaccine",
            "final_score": 0.88,
            "credibility": 0.95,
        },
    ]
    llm_result = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.8,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vaccines cause autism",
                "status": "INVALID",
                "supporting_fact": "Childhood vaccines do not cause autism.",
                "source_url": "https://who.int/autism",
            },
            {
                "claim_segment": "Vaccines cause the flu",
                "status": "INVALID",
                "supporting_fact": "Flu vaccines do not cause flu illness.",
                "source_url": "https://cdc.gov/flu-vaccine",
            },
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)
    assert parsed["verdict"] == "FALSE"
    assert parsed["required_segments_resolved"] is True


def test_all_unknown_maps_to_unverifiable():
    vg = _vg()
    claim = "Vaccines do not cause autism or the flu."
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vaccines do not cause autism",
                "status": "UNKNOWN",
                "supporting_fact": "",
                "source_url": "",
            },
            {
                "claim_segment": "Vaccines do not cause the flu",
                "status": "UNKNOWN",
                "supporting_fact": "",
                "source_url": "",
            },
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, [])
    assert parsed["verdict"] == "UNVERIFIABLE"
    assert parsed["required_segments_resolved"] is False


def test_combined_valid_breakdown_is_rebuilt_to_two_subclaims_and_not_true():
    vg = _vg()
    claim = "Vaccines do not cause autism or the flu."
    evidence = [
        {
            "statement": "The data show that vaccines do not cause autism.",
            "source_url": "https://www.statnews.com/example",
            "final_score": 0.73,
            "credibility": 0.9,
        }
    ]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vaccines do not cause autism or the flu",
                "status": "VALID",
                "supporting_fact": "The data show that vaccines do not cause autism.",
                "source_url": "https://www.statnews.com/example",
            }
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)
    statuses = [str(item.get("status", "UNKNOWN")).upper() for item in parsed["claim_breakdown"]]

    assert len(parsed["claim_breakdown"]) == 2
    assert parsed["verdict"] == "PARTIALLY_TRUE"
    assert parsed["required_segments_resolved"] is False
    assert "UNKNOWN" in statuses
    assert "unresolved" in parsed["rationale"].lower()


def test_flu_segment_cannot_be_marked_valid_from_autism_only_evidence():
    vg = _vg()
    claim = "Vaccines do not cause autism or the flu."
    evidence = [
        {
            "statement": "The data show that vaccines do not cause autism.",
            "source_url": "https://www.statnews.com/example",
            "final_score": 0.73,
            "credibility": 0.9,
        }
    ]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vaccines do not cause autism",
                "status": "VALID",
                "supporting_fact": "The data show that vaccines do not cause autism.",
                "source_url": "https://www.statnews.com/example",
            },
            {
                "claim_segment": "Vaccines do not cause the flu",
                "status": "VALID",
                "supporting_fact": "The data show that vaccines do not cause autism.",
                "source_url": "https://www.statnews.com/example",
            },
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)
    flu_item = next(
        item
        for item in parsed["claim_breakdown"]
        if "flu" in str(item.get("claim_segment", "")).lower()
        or "influenza" in str(item.get("claim_segment", "")).lower()
    )
    assert str(flu_item.get("status", "")).upper() == "UNKNOWN"


def test_valid_segment_is_flipped_when_supporting_fact_is_may_cause():
    vg = _vg()
    claim = "Vaccines do not cause autism."
    evidence = [
        {
            "statement": "Some studies suggest vaccines may cause autism in children.",
            "source_url": "https://example.org/contradict",
            "final_score": 0.9,
            "credibility": 0.9,
        }
    ]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vaccines do not cause autism",
                "status": "VALID",
                "supporting_fact": "Some studies suggest vaccines may cause autism in children.",
                "source_url": "https://example.org/contradict",
            }
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)
    assert parsed["claim_breakdown"][0]["status"] in {"INVALID", "PARTIALLY_INVALID"}
    assert parsed["verdict"] in {"FALSE", "PARTIALLY_TRUE"}


def test_confidence_is_capped_when_subclaims_unresolved_and_policy_not_sufficient():
    vg = _vg()
    claim = "Vaccines do not cause autism or the flu."
    evidence = [
        {
            "statement": "Childhood vaccines do not cause autism.",
            "source_url": "https://who.int/autism",
            "final_score": 0.93,
            "credibility": 0.95,
        }
    ]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.98,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vaccines do not cause autism",
                "status": "VALID",
                "supporting_fact": "Childhood vaccines do not cause autism.",
                "source_url": "https://who.int/autism",
            }
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)
    assert parsed["unresolved_segments"] >= 1
    assert parsed["policy_sufficient"] is False
    assert parsed["confidence"] <= 0.56
