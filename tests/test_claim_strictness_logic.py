from app.services.logic.claim_strictness import compute_claim_strictness


def test_claim_strictness_high_for_assertive_universal_claim():
    p = compute_claim_strictness("This treatment always prevents infection in all adults.")
    assert p.required_evidence_level in {"HIGH", "VERY_HIGH"}
    assert p.assertiveness_score >= 0.5
    assert p.universality_score >= 0.5


def test_claim_strictness_lower_for_probabilistic_claim():
    p = compute_claim_strictness("This treatment may reduce risk in some people under certain conditions.")
    assert p.required_evidence_level in {"LOW", "MEDIUM"}
    assert p.modality_score < 0.75


def test_claim_type_numeric_comparative_detected():
    p = compute_claim_strictness("Group A has a 20% lower risk than Group B.")
    assert p.claim_type in {"NUMERIC", "COMPARATIVE", "RISK_ASSOCIATION"}
    assert p.falsifiability_score > 0.3
