from app.services.corrective.fact_extractor import FactExtractor
from app.services.verdict.v2.calibration import ConfidenceCalibrator
from app.services.verdict.verdict_generator import VerdictGenerator


def test_calibrator_distribution_keeps_directional_refute_from_becoming_unverifiable():
    calibrator = ConfidenceCalibrator(None)
    probs = calibrator.calibrate_distribution(
        {"true": 0.05, "false": 0.43, "unverifiable": 0.52},
        features={
            "coverage": 0.10,
            "admissible_ratio": 1.0,
            "contradict_signal": 0.79,
            "support_signal": 0.10,
        },
    )
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["false"] > probs["unverifiable"]


def test_soft_align_class_probs_with_verdict_preserves_uncertainty():
    aligned = VerdictGenerator._soft_align_class_probs_with_verdict(
        {"true": 0.05, "false": 0.43, "unverifiable": 0.52},
        verdict="FALSE",
    )
    assert aligned["false"] > aligned["unverifiable"]
    assert aligned["unverifiable"] >= 0.20
    assert abs(sum(aligned.values()) - 1.0) < 1e-6


def test_fact_extractor_deterministic_fallback_extracts_claim_relevant_sentence():
    fe = FactExtractor.__new__(FactExtractor)
    fe._generic_anchor_tokens = {
        "health",
        "healthy",
        "disease",
        "diseases",
        "condition",
        "conditions",
        "symptom",
        "symptoms",
        "vitamin",
        "supplement",
        "supplements",
        "immune",
        "immunity",
    }
    fe._grounding_stop_tokens = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "to",
        "for",
        "of",
        "in",
        "on",
        "with",
        "by",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
    }
    pages = [
        {
            "url": "https://example.org/tobacco",
            "source": "example.org",
            "published_at": "2025-01-01",
            "content": (
                "Tobacco use remains a leading preventable cause of death. "
                "Quitting smoking lowers lung cancer risk over time. "
                "Exercise is also important for wellbeing."
            ),
        }
    ]
    facts = fe._deterministic_sentence_fallback(
        pages,
        claim_text="Reducing tobacco use lowers lung cancer risk",
        claim_entities=["tobacco use", "lung cancer"],
        must_have_entities=["tobacco", "lung cancer"],
    )
    assert facts
    assert any("lung cancer risk" in str(f.get("statement", "")).lower() for f in facts)
