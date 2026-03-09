from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)
    vg._last_predicate_queries_generated = []
    return vg


def test_risk_direction_conflict_is_not_admitted_as_support():
    vg = _vg()
    claim = "Physical activity increases the risk of heart disease in adults."
    statement = (
        "Adults following healthy diet and physical activity guidance have lower "
        "cardiovascular morbidity and mortality."
    )
    evidence = [
        {
            "statement": statement,
            "source_url": "https://example.org/cardio",
            "final_score": 0.76,
            "credibility": 0.95,
        }
    ]
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": statement,
            "relevance": "NEUTRAL",
            "relevance_score": 0.76,
            "source_url": "https://example.org/cardio",
        }
    ]
    normalized = vg._normalize_evidence_map(claim, evidence_map, evidence)
    assert normalized
    assert str(normalized[0].get("relevance") or "").upper() != "SUPPORTS"


def test_neutral_only_trust_gate_low_margin_uses_sigmoid_tiebreak():
    vg = _vg()
    claim = "Vaccination coverage disruptions increase outbreak risk."
    payload = {
        "verdict": "UNVERIFIABLE",
        "truthfulness_percent": 45.0,
        "confidence": 0.5,
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN", "evidence_used_ids": []}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Global vaccination status summary.",
                "relevance": "NEUTRAL",
                "relevance_score": 0.31,
                "source_url": "https://example.org/vax",
            }
        ],
        "support_mass": 0.29,
        "contradict_mass": 0.25,
        "class_probs": {"true": 0.37, "false": 0.35, "unverifiable": 0.28},
        "trust_threshold_met": False,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["verdict_binary"] == "FALSE"
    assert out.get("binary_collapse_reason") == "abstain_to_false_policy"


def test_inverse_modifier_claim_maps_protective_statement_to_support():
    vg = _vg()
    claim = "Vaccination coverage disruptions increase outbreak risk."
    statement = "Vaccination coverage is critical for preventing outbreaks."
    evidence = [
        {
            "statement": statement,
            "source_url": "https://example.org/vax",
            "final_score": 0.81,
            "credibility": 0.95,
        }
    ]
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": statement,
            "relevance": "NEUTRAL",
            "relevance_score": 0.81,
            "source_url": "https://example.org/vax",
        }
    ]
    normalized = vg._normalize_evidence_map(claim, evidence_map, evidence)
    assert normalized
    assert str(normalized[0].get("relevance") or "").upper() == "SUPPORTS"


def test_effect_direction_detects_lead_to_as_harm_direction():
    assert VerdictGenerator._effect_direction_sign("Coverage disruptions can lead to outbreaks.") == 1
