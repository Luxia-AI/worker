from app.services.verdict.verdict_generator import VerdictGenerator


def test_display_verdict_preserves_partial_when_trust_failed():
    vg = VerdictGenerator.__new__(VerdictGenerator)
    out = vg._display_verdict_label(
        verdict="PARTIALLY_TRUE",
        truthfulness_percent=55.0,
        trust_gate_passed=False,
        analysis_counts={"map_support_signal_max": 0.7, "map_contradict_signal_max": 0.1},
    )
    assert out == "PARTIALLY_TRUE"
