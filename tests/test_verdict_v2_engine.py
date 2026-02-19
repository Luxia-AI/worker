from app.services.verdict.v2.calibration import ConfidenceCalibrator
from app.services.verdict.v2.normalizer import is_blocked_content
from app.services.verdict.v2.reconciler import reconcile_verdict
from app.services.verdict.v2.shadow import compute_shadow_diff


def test_reconciler_partial_support_does_not_emit_true():
    decision = reconcile_verdict(["VALID", "PARTIALLY_VALID"])
    assert decision.verdict == "PARTIALLY_TRUE"
    assert decision.required_segments_resolved is True


def test_blocked_content_patterns_are_detected():
    assert is_blocked_content("Access denied to the URL")
    assert is_blocked_content("permission required to access the URL")
    assert not is_blocked_content("Vitamin C supports immune function.")


def test_calibrator_fallback_without_artifact():
    calibrator = ConfidenceCalibrator("worker/tests/fixtures/no_such_calibrator.json")
    assert calibrator.version is None
    assert calibrator.calibrate(0.73, {"coverage": 1.0}) == 0.73


def test_shadow_diff_has_schema_and_diffs():
    v1 = {"verdict": "TRUE", "confidence": 0.9, "truthfulness_percent": 95.0, "claim_breakdown": [{"status": "VALID"}]}
    v2 = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.7,
        "truthfulness_percent": 75.0,
        "claim_breakdown": [{"status": "PARTIALLY_VALID"}],
    }
    out = compute_shadow_diff(v1, v2)
    assert out["parity"] is False
    assert "verdict" in out["diffs"]
    assert "v1_schema_keys" in out
    assert "v2_schema_keys" in out
