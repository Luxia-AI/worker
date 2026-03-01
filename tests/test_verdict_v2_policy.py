from app.services.verdict.v2.calibration import ConfidenceCalibrator
from app.services.verdict.v2.policy import compute_verdict_policy_v2
from app.services.verdict.v2.stance_pipeline import build_evidence_scores_v2
from app.services.verdict.v2.types import EvidenceScoreV2


def test_verdict_policy_v2_is_deterministic_for_identical_inputs():
    scores = [
        EvidenceScoreV2(
            evidence_id=0,
            support_score=0.05,
            contradict_score=0.86,
            neutral_score=0.09,
            nli_entail_prob=0.05,
            nli_contradict_prob=0.86,
            nli_neutral_prob=0.09,
            admissible=True,
            weight=0.92,
            source_domain="nih.gov",
        ),
        EvidenceScoreV2(
            evidence_id=1,
            support_score=0.07,
            contradict_score=0.79,
            neutral_score=0.14,
            nli_entail_prob=0.07,
            nli_contradict_prob=0.79,
            nli_neutral_prob=0.14,
            admissible=True,
            weight=0.88,
            source_domain="who.int",
        ),
    ]
    calibrator = ConfidenceCalibrator(None)
    out1 = compute_verdict_policy_v2(scores, coverage=0.95, diversity=0.75, calibrator=calibrator)
    out2 = compute_verdict_policy_v2(scores, coverage=0.95, diversity=0.75, calibrator=calibrator)

    assert out1 == out2
    assert out1["verdict"] == "FALSE"


def test_verdict_policy_v2_caps_unverifiable_confidence_under_low_signal():
    scores = [
        EvidenceScoreV2(
            evidence_id=0,
            support_score=0.26,
            contradict_score=0.24,
            neutral_score=0.50,
            nli_entail_prob=0.26,
            nli_contradict_prob=0.24,
            nli_neutral_prob=0.50,
            admissible=True,
            weight=0.15,
            source_domain="example.org",
        )
    ]
    calibrator = ConfidenceCalibrator(None)
    out = compute_verdict_policy_v2(scores, coverage=0.12, diversity=0.18, calibrator=calibrator)
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["calibrated_confidence"] <= 0.60


def test_stance_pipeline_emits_nonzero_refute_admission_stats(monkeypatch):
    monkeypatch.setenv("REFUTE_NLI_ENABLED", "false")
    claim = "Omega-3 supplements do not reduce inflammation"
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": "Omega-3 supplements reduce inflammation markers.",
            "relevance": "REFUTES",
            "relevance_score": 0.92,
            "contradiction_score": 0.88,
            "predicate_match_score": 0.95,
            "credibility": 0.9,
            "scope_alignment": 1.0,
        },
        {
            "evidence_id": 1,
            "statement": "Some trials found mixed effects.",
            "relevance": "NEUTRAL",
            "relevance_score": 0.40,
            "credibility": 0.6,
            "scope_alignment": 0.8,
        },
    ]
    evidence = [
        {"statement": "Omega-3 supplements reduce inflammation markers.", "source_url": "https://nih.gov/e1"},
        {"statement": "Some trials found mixed effects.", "source_url": "https://example.org/e2"},
    ]
    scores, diag = build_evidence_scores_v2(claim=claim, evidence_map=evidence_map, evidence=evidence, nli_top_n=2)
    assert len(scores) == 2
    assert diag["refute_candidate_count_stage1"] >= 1
    assert diag["refutes_admission_rate"] > 0.0
