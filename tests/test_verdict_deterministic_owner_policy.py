from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)
    vg._last_predicate_queries_generated = []
    return vg


def test_internal_unverifiable_always_collapses_binary_to_false():
    vg = _vg()
    claim = "Intervention X improves condition Y."
    payload = {
        "verdict": "UNVERIFIABLE",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN", "evidence_used_ids": []}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Background context with no directional conclusion.",
                "relevance": "NEUTRAL",
                "relevance_score": 0.42,
                "source_url": "https://example.org/context",
            }
        ],
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict_internal"] == "UNVERIFIABLE"
    assert out["verdict_binary"] == "FALSE"
    assert out["binary_collapse_reason"] == "abstain_to_false_policy"
    assert bool(out.get("verdict_field_invariant_passed", False))


def test_neutral_mass_alone_cannot_produce_directional_internal_verdict():
    vg = _vg()
    claim = "Supplement X prevents all infections."
    payload = {
        "verdict": "TRUE",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN", "evidence_used_ids": []}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "General discussion of supplement markets.",
                "relevance": "NEUTRAL",
                "relevance_score": 0.70,
                "source_url": "https://example.org/neutral",
            }
        ],
        "sufficiency_score": 0.95,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict_internal"] == "UNVERIFIABLE"
    assert out["verdict_binary"] == "FALSE"


def test_refute_gate_fields_persisted_in_normalized_evidence():
    vg = _vg()
    claim = "Vaccines contain microchips."
    statement = "COVID-19 vaccines do not contain microchips or tracking devices."
    evidence = [
        {
            "statement": statement,
            "source_url": "https://example.org/cdc",
            "final_score": 0.85,
            "credibility": 0.95,
        }
    ]
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": statement,
            "relevance": "NEUTRAL",
            "relevance_score": 0.85,
            "source_url": "https://example.org/cdc",
        }
    ]
    normalized = vg._normalize_evidence_map(claim, evidence_map, evidence)
    assert normalized
    row = normalized[0]
    assert "refute_eligibility_score" in row
    assert "refute_threshold" in row
    assert "refute_gate_passed" in row
    assert "refute_gate_block_reason" in row


def test_contradiction_with_sufficient_signal_yields_false_internal_verdict():
    vg = _vg()
    claim = "Alcohol has never caused a single death."
    payload = {
        "verdict": "UNVERIFIABLE",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN", "evidence_used_ids": []}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Mortality attributable to alcohol consumption is documented globally.",
                "relevance": "REFUTES",
                "relevance_score": 0.91,
                "refute_score": 0.90,
                "refute_gate_passed": True,
                "refute_eligibility_score": 0.90,
                "refute_threshold": 0.60,
                "credibility": 0.95,
                "source_url": "https://example.org/gbd",
            }
        ],
        "sufficiency_score": 0.92,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict_internal"] == "FALSE"
    assert out["verdict_binary"] == "FALSE"
