from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)
    vg._last_predicate_queries_generated = []
    return vg


def test_binary_false_without_refute_signal_downgrades_to_unverifiable():
    vg = _vg()
    claim = "X has no association with Y"
    payload = {
        "verdict": "FALSE",
        "truthfulness_percent": 18.0,
        "confidence": 0.84,
        "claim_breakdown": [
            {
                "claim_segment": claim,
                "status": "INVALID",
                "supporting_fact": "X is associated with Y in observational cohorts.",
                "source_url": "https://example.org/source",
                "evidence_used_ids": [0],
            }
        ],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "X is associated with Y in observational cohorts.",
                "relevance": "SUPPORTS",
                "relevance_score": 0.78,
                "contradiction_score": 0.08,
                "nli_contradict_prob": 0.05,
                "object_match_ok": False,
                "predicate_match_score": 0.28,
                "source_url": "https://example.org/source",
            }
        ],
        "class_probs": {"true": 0.46, "false": 0.28, "unverifiable": 0.26},
        "trust_threshold_met": True,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["verdict_binary"] == "FALSE"
    assert bool(out.get("verdict_field_invariant_passed", False))


def test_binary_true_without_support_signal_downgrades_to_unverifiable():
    vg = _vg()
    claim = "X prevents Y"
    payload = {
        "verdict": "TRUE",
        "truthfulness_percent": 88.0,
        "confidence": 0.86,
        "claim_breakdown": [
            {
                "claim_segment": claim,
                "status": "VALID",
                "supporting_fact": "No conclusive data demonstrates prevention.",
                "source_url": "https://example.org/source",
                "evidence_used_ids": [0],
            }
        ],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "No conclusive data demonstrates prevention.",
                "relevance": "REFUTES",
                "relevance_score": 0.74,
                "contradiction_score": 0.72,
                "nli_contradict_prob": 0.70,
                "object_match_ok": True,
                "predicate_match_score": 0.52,
                "source_url": "https://example.org/source",
            }
        ],
        "class_probs": {"true": 0.33, "false": 0.41, "unverifiable": 0.26},
        "trust_threshold_met": True,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict"] == "UNVERIFIABLE"


def test_neutral_only_trust_gate_uses_directional_signal_for_binary_projection():
    vg = _vg()
    claim = "X can improve Y in adults"
    payload = {
        "verdict": "UNVERIFIABLE",
        "truthfulness_percent": 44.0,
        "confidence": 0.5,
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN", "evidence_used_ids": []}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Contextual background statement about X and Y.",
                "relevance": "NEUTRAL",
                "relevance_score": 0.42,
                "source_url": "https://example.org/source",
            }
        ],
        "support_mass": 0.24,
        "contradict_mass": 0.05,
        "class_probs": {"true": 0.58, "false": 0.30, "unverifiable": 0.12},
        "trust_threshold_met": False,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["verdict_binary"] == "FALSE"
    assert out.get("binary_collapse_reason") == "abstain_to_false_policy"


def test_contradicts_label_is_counted_as_directional_refute_signal():
    vg = _vg()
    claim = "Daily supplement X prevents all respiratory infections."
    payload = {
        "verdict": "FALSE",
        "truthfulness_percent": 20.0,
        "confidence": 0.64,
        "claim_breakdown": [
            {
                "claim_segment": claim,
                "status": "INVALID",
                "supporting_fact": "Evidence shows no universal prevention effect.",
                "source_url": "https://example.org/source",
                "evidence_used_ids": [0],
            }
        ],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Evidence shows no universal prevention effect.",
                "relevance": "CONTRADICTS",
                "relevance_score": 0.82,
                "refute_score": 0.86,
                "refute_gate_passed": True,
                "refute_eligibility_score": 0.86,
                "refute_threshold": 0.60,
                "contradiction_score": 0.10,
                "nli_contradict_prob": 0.10,
                "object_match_ok": True,
                "predicate_match_score": 0.36,
                "credibility": 0.92,
                "source_url": "https://example.org/source",
            }
        ],
        "class_probs": {"true": 0.22, "false": 0.54, "unverifiable": 0.24},
        "trust_threshold_met": True,
        "sufficiency_score": 0.92,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict"] == "FALSE"
    assert out["verdict_binary"] == "FALSE"


def test_low_delta_binary_projection_uses_sigmoid_instead_of_default_false():
    vg = _vg()
    claim = "Intervention X has mixed evidence on reducing endpoint Y."
    payload = {
        "verdict": "UNVERIFIABLE",
        "truthfulness_percent": 50.0,
        "confidence": 0.5,
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN", "evidence_used_ids": []}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Some studies report endpoint reduction under intervention X.",
                "relevance": "SUPPORTS",
                "relevance_score": 0.41,
                "source_url": "https://example.org/s1",
            },
            {
                "evidence_id": 1,
                "statement": "Other studies find no consistent endpoint reduction.",
                "relevance": "CONTRADICTS",
                "relevance_score": 0.39,
                "source_url": "https://example.org/s2",
            },
        ],
        "support_mass": 0.41,
        "contradict_mass": 0.39,
        "class_probs": {"true": 0.36, "false": 0.35, "unverifiable": 0.29},
        "trust_threshold_met": True,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["verdict_binary"] == "FALSE"
    assert out.get("binary_collapse_reason") == "abstain_to_false_policy"


def test_binary_projection_uses_normalized_evidence_map_masses_over_stale_payload_masses():
    vg = _vg()
    claim = "Intervention X reduces endpoint Y."
    payload = {
        "verdict": "UNVERIFIABLE",
        "truthfulness_percent": 50.0,
        "confidence": 0.5,
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN", "evidence_used_ids": []}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Intervention X reduces endpoint Y in controlled studies.",
                "relevance": "SUPPORTS",
                "relevance_score": 0.82,
                "support_score": 0.82,
                "credibility": 0.90,
                "source_url": "https://example.org/s1",
            }
        ],
        # Simulate stale masses from a pre-normalization stage.
        "support_mass": 0.10,
        "contradict_mass": 0.90,
        "neutral_mass": 0.0,
        "class_probs": {"true": 0.42, "false": 0.33, "unverifiable": 0.25},
        "trust_threshold_met": True,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["support_mass"] > out["contradict_mass"]
    assert out["verdict_binary"] == "FALSE"
    assert out["verdict"] == "UNVERIFIABLE"


def test_binary_projection_keeps_truth_and_confidence_consistent_when_internal_unverifiable():
    vg = _vg()
    claim = "Disruptions in intervention X increase risk of outcome Y."
    payload = {
        "verdict": "UNVERIFIABLE",
        "truthfulness_percent": 44.0,
        "confidence": 0.30,
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN", "evidence_used_ids": []}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Disruptions in intervention X can lead to higher outcome Y risk.",
                "relevance": "SUPPORTS",
                "relevance_score": 0.85,
                "source_url": "https://example.org/s1",
            }
        ],
        "class_probs": {"true": 0.58, "false": 0.17, "unverifiable": 0.25},
        "trust_threshold_met": True,
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=[])
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["verdict_binary"] == "FALSE"
    assert float(out["truthfulness_percent"]) <= 65.0
    assert float(out["confidence"]) <= 0.55
