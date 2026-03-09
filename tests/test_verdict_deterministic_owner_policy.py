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


def test_support_labeled_evidence_is_admitted_to_support_mass():
    vg = _vg()
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": "WHO recommends reducing junk food intake.",
            "relevance": "SUPPORTS",
            "relevance_score": 0.78,
            "support_score": 0.82,
            "predicate_match_score": 0.72,
            "subject_match_score": 0.70,
            "object_match_score": 0.75,
            "anchor_match_score": 0.70,
            "credibility": 0.90,
            "source_url": "https://example.org/guidance",
        }
    ]
    support_mass, _, _, _, diagnostics = vg._compute_deterministic_masses(
        claim="WHO recommends reducing junk food intake.",
        evidence_map=evidence_map,
        evidence_count=1,
        source_domains=1,
    )
    assert diagnostics["support_labeled_count"] > 0
    assert diagnostics["support_admitted_count"] > 0
    assert support_mass > 0.0
    assert bool(evidence_map[0].get("admission_passed", False))
    assert bool(evidence_map[0].get("admitted_to_support_mass", False))


def test_refute_gate_fields_populated_for_refute_candidate():
    vg = _vg()
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": "WHO advises against daily junk food consumption.",
            "relevance": "REFUTES",
            "relevance_score": 0.74,
            "refute_score": 0.66,
            "contradiction_score": 0.68,
            "predicate_match_score": 0.55,
            "subject_match_score": 0.60,
            "object_match_score": 0.59,
            "anchor_match_score": 0.62,
            "credibility": 0.88,
            "source_url": "https://example.org/guidance",
        }
    ]
    _, contradict_mass, _, _, _ = vg._compute_deterministic_masses(
        claim="WHO recommends eating junk food daily.",
        evidence_map=evidence_map,
        evidence_count=1,
        source_domains=1,
    )
    row = evidence_map[0]
    assert float(row.get("refute_eligibility_score", 0.0) or 0.0) > 0.0
    assert float(row.get("refute_threshold", 0.0) or 0.0) > 0.0
    assert "refute_gate_passed" in row
    if not bool(row.get("refute_gate_passed", False)):
        assert str(row.get("refute_gate_block_reason") or "").strip()
    assert contradict_mass >= 0.0


def test_labeled_support_with_zero_mass_emits_diagnostic_warning():
    vg = _vg()
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": "Weakly related statement.",
            "relevance": "SUPPORTS",
            "relevance_score": 0.10,
            "support_score": 0.10,
            "predicate_match_score": 0.0,
            "subject_match_score": 0.0,
            "object_match_score": 0.0,
            "anchor_match_score": 0.0,
            "credibility": 0.10,
            "source_url": "https://example.org/weak",
        }
    ]
    support_mass, _, _, _, diagnostics = vg._compute_deterministic_masses(
        claim="Any health claim.",
        evidence_map=evidence_map,
        evidence_count=1,
        source_domains=1,
    )
    assert diagnostics["support_labeled_count"] > 0
    assert support_mass == 0.0
    assert diagnostics["support_zero_with_labels"] is True
    assert "support_labeled_but_zero_support_mass" in diagnostics["warnings"]
    assert any(str(reason).strip() for reason in diagnostics["admission_block_reasons"])
