from app.services.logic.claim_strictness import compute_claim_strictness
from app.services.logic.evidence_strength import EvidenceStrength
from app.services.logic.overrides import apply_claim_logic_overrides


def test_contradiction_dominance_override_for_high_strictness_claim():
    strictness = compute_claim_strictness("This treatment always prevents severe disease in all adults.")
    evidence_strengths = [
        EvidenceStrength(0.30, 0.60, 0.50, 0.75, "SUPPORTS", 0.0),
        EvidenceStrength(0.85, 0.05, 0.02, 0.95, "REFUTES", 0.0),
        EvidenceStrength(0.80, 0.05, 0.01, 0.93, "REFUTES", 0.0),
        EvidenceStrength(0.79, 0.08, 0.02, 0.90, "REFUTES", 0.0),
    ]
    breakdown = [
        {"status": "INVALID"},
        {"status": "PARTIALLY_INVALID"},
        {"status": "UNKNOWN"},
    ]
    out = apply_claim_logic_overrides(
        claim="x",
        strictness=strictness,
        evidence_strengths=evidence_strengths,
        claim_breakdown=breakdown,
        verdict="TRUE",
        truthfulness_percent=76.0,
        confidence=0.9,
        diversity=0.7,
        agreement=0.8,
        evidence_count=4,
        kg_hint_ratio=0.0,
    )
    assert out.override_fired == "CONTRADICTION_DOMINANCE"
    assert out.verdict == "FALSE"
    assert out.truthfulness_cap_percent is not None and out.truthfulness_cap_percent <= 20.0


def test_diversity_confidence_cap_applies():
    strictness = compute_claim_strictness("This treatment always cures disease.")
    out = apply_claim_logic_overrides(
        claim="x",
        strictness=strictness,
        evidence_strengths=[EvidenceStrength(0.9, 0.0, 0.0, 0.95, "SUPPORTS", 0.0)],
        claim_breakdown=[{"status": "VALID"}],
        verdict="TRUE",
        truthfulness_percent=95.0,
        confidence=0.95,
        diversity=0.0,
        agreement=1.0,
        evidence_count=1,
        kg_hint_ratio=0.0,
    )
    assert out.confidence_cap is not None
    assert out.confidence_cap <= 0.75


def test_multihop_relaxation_to_partially_true():
    strictness = compute_claim_strictness(
        "If a patient improves diet and exercise, medication dose can sometimes be reduced."
    )
    out = apply_claim_logic_overrides(
        claim="x",
        strictness=strictness,
        evidence_strengths=[EvidenceStrength(0.7, 0.1, 0.05, 0.9, "SUPPORTS", 0.0)],
        claim_breakdown=[{"status": "VALID"}, {"status": "UNKNOWN"}],
        verdict="UNVERIFIABLE",
        truthfulness_percent=42.0,
        confidence=0.55,
        diversity=0.5,
        agreement=0.85,
        evidence_count=3,
        kg_hint_ratio=0.2,
    )
    assert out.override_fired in {"MULTIHOP_RELAXATION", "DIVERSITY_CAP"}
    if out.override_fired == "MULTIHOP_RELAXATION":
        assert out.verdict == "PARTIALLY_TRUE"
