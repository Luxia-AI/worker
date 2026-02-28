from app.services.verdict.v2.calibration import ConfidenceCalibrator


def test_calibrator_feature_penalty_for_contradiction():
    calibrator = ConfidenceCalibrator(None)
    hi = calibrator.calibrate(
        0.8,
        features={
            "coverage": 0.9,
            "agreement": 0.9,
            "diversity": 0.8,
            "contradict_signal": 0.0,
            "admissible_ratio": 0.9,
            "evidence_quality": 0.9,
        },
    )
    lo = calibrator.calibrate(
        0.8,
        features={
            "coverage": 0.4,
            "agreement": 0.4,
            "diversity": 0.3,
            "contradict_signal": 0.8,
            "admissible_ratio": 0.5,
            "evidence_quality": 0.4,
        },
    )
    assert hi > lo


def test_calibrator_distribution_prefers_unverifiable_when_sparse():
    calibrator = ConfidenceCalibrator(None)
    probs = calibrator.calibrate_distribution(
        {"true": 0.4, "false": 0.3, "unverifiable": 0.3},
        features={"coverage": 0.2, "admissible_ratio": 0.3, "contradict_signal": 0.1, "support_signal": 0.1},
    )
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["unverifiable"] >= 0.3
