from app.services.domain import HealthScopeGate


def test_health_scope_gate_detects_biomedical_claim():
    gate = HealthScopeGate(min_confidence=0.45)
    out = gate.classify(
        "A randomized clinical trial evaluated treatment dose and patient outcomes in a cohort study.",
        declared_domain="health",
    )
    assert out.health_in_scope is True
    assert out.biomedical_confidence >= 0.45


def test_health_scope_gate_blocks_generic_non_health_claim():
    gate = HealthScopeGate(min_confidence=0.45)
    out = gate.classify(
        "This policy discussion focuses on market regulation and transportation costs.",
        declared_domain="general",
    )
    assert out.health_in_scope is False
    assert out.biomedical_confidence < 0.45
