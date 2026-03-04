from app.services.verdict.v2.calibration import ConfidenceCalibrator


def test_calibrate_distribution_does_not_over_amplify_unverifiable():
    """With v3.3-like inputs, UNVERIFIABLE probability should not reach 0.93+."""
    calibrator = ConfidenceCalibrator(None)
    probs = calibrator.calibrate_distribution(
        {"true": 0.07, "false": 0.08, "unverifiable": 0.85},
        features={
            "coverage": 0.35,
            "admissible_ratio": 1.0,
            "contradict_signal": 0.21,
            "support_signal": 0.18,
        },
    )
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["unverifiable"] < 0.88, (
        f"UNVERIFIABLE {probs['unverifiable']:.4f} should not be amplified above 0.88 "
        f"for moderate coverage with some directional signal"
    )


def test_calibrate_distribution_preserves_directional_signal():
    """When support clearly dominates contradict, calibration should not erase the difference."""
    calibrator = ConfidenceCalibrator(None)
    probs = calibrator.calibrate_distribution(
        {"true": 0.50, "false": 0.15, "unverifiable": 0.35},
        features={
            "coverage": 0.60,
            "admissible_ratio": 0.80,
            "contradict_signal": 0.10,
            "support_signal": 0.55,
        },
    )
    assert probs["true"] > probs["false"], (
        f"Support-dominant input should preserve true > false in output; "
        f"got true={probs['true']:.4f}, false={probs['false']:.4f}"
    )
    assert probs["true"] > probs["unverifiable"], (
        f"Support-dominant input with good coverage should not flip to unverifiable; "
        f"got true={probs['true']:.4f}, unverifiable={probs['unverifiable']:.4f}"
    )


def test_calibrate_distribution_still_boosts_unverifiable_when_truly_sparse():
    """Very low coverage and admissibility should still push UNVERIFIABLE up."""
    calibrator = ConfidenceCalibrator(None)
    probs = calibrator.calibrate_distribution(
        {"true": 0.40, "false": 0.30, "unverifiable": 0.30},
        features={
            "coverage": 0.10,
            "admissible_ratio": 0.20,
            "contradict_signal": 0.05,
            "support_signal": 0.05,
        },
    )
    assert (
        probs["unverifiable"] >= 0.30
    ), f"Sparse evidence should still boost UNVERIFIABLE; got {probs['unverifiable']:.4f}"


def test_calibrate_distribution_weak_polarity_penalty_only_at_very_low_strength():
    """The weak-polarity penalty should only activate when polarity_strength <= 0.30."""
    calibrator = ConfidenceCalibrator(None)
    # polarity_strength = 0.40 (should NOT trigger penalty)
    probs_moderate = calibrator.calibrate_distribution(
        {"true": 0.35, "false": 0.30, "unverifiable": 0.35},
        features={
            "coverage": 0.50,
            "admissible_ratio": 0.60,
            "contradict_signal": 0.40,
            "support_signal": 0.20,
        },
    )
    # polarity_strength = 0.15 (should trigger penalty)
    probs_weak = calibrator.calibrate_distribution(
        {"true": 0.35, "false": 0.30, "unverifiable": 0.35},
        features={
            "coverage": 0.50,
            "admissible_ratio": 0.60,
            "contradict_signal": 0.15,
            "support_signal": 0.10,
        },
    )
    # The weak-polarity case should have higher UNVERIFIABLE than moderate
    assert probs_weak["unverifiable"] >= probs_moderate["unverifiable"], (
        f"Weak polarity should yield higher UNVERIFIABLE; "
        f"weak={probs_weak['unverifiable']:.4f}, moderate={probs_moderate['unverifiable']:.4f}"
    )
