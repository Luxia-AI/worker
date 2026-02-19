from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    return VerdictGenerator.__new__(VerdictGenerator)


def test_segment_recovery_hints_expand_for_iron_fatigue_claims():
    vg = _vg()
    segment = "Iron contributes to the reduction of tiredness and fatigue"
    hints = [h.lower() for h in vg._segment_recovery_query_hints(segment)]
    joined = " | ".join(hints)
    assert "iron supplementation fatigue randomized trial" in joined
    assert "iron deficiency fatigue improvement evidence" in joined
    assert "reduction of tiredness and fatigue" in joined
