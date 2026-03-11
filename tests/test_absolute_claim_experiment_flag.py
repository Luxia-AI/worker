import copy

from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)
    vg._last_predicate_queries_generated = []
    return vg


def _normalize(vg: VerdictGenerator, claim: str, statement: str):
    return vg._normalize_evidence_map(
        claim=claim,
        evidence_map=[
            {
                "evidence_id": 0,
                "statement": statement,
                "relevance": "NEUTRAL",
                "relevance_score": 0.78,
                "source_url": "https://example.org/evidence",
            }
        ],
        evidence=[
            {
                "statement": statement,
                "source_url": "https://example.org/evidence",
                "final_score": 0.78,
                "credibility": 0.9,
            }
        ],
    )[0]


def test_absolute_experiment_off_does_not_use_absolute_promotion(monkeypatch):
    monkeypatch.setenv("ENABLE_ABSOLUTE_CLAIM_CONTRADICTION_EXPERIMENT", "false")
    vg = _vg()
    row = _normalize(
        vg,
        "Vitamin C completely prevents the common cold.",
        "Vitamin C reduces duration and severity of common cold symptoms.",
    )
    assert bool(row.get("absolute_claim_experiment_enabled", False)) is False
    assert bool(row.get("absolute_claim_experiment_path_used", False)) is False
    assert "absolute_claim_non_universal_contradiction" not in str(row.get("refute_candidate_reason") or "")


def test_absolute_experiment_on_can_activate_absolute_promotion(monkeypatch):
    monkeypatch.setenv("ENABLE_ABSOLUTE_CLAIM_CONTRADICTION_EXPERIMENT", "true")
    vg = _vg()
    row = _normalize(
        vg,
        "Vitamin C completely prevents the common cold.",
        "Vitamin C reduces duration and severity of common cold symptoms.",
    )
    assert bool(row.get("absolute_claim_experiment_enabled", False))
    assert bool(row.get("absolute_claim_experiment_path_used", False))
    assert bool(row.get("contradiction_to_absolute_flag", False))
    assert "absolute_claim_non_universal_contradiction" in str(row.get("refute_candidate_reason") or "")


def test_experiment_off_preserves_non_absolute_semantics(monkeypatch):
    claim = "Omega-3 supplements reduce inflammation."
    statement = "Omega-3 supplements reduce inflammation markers in trials."

    monkeypatch.setenv("ENABLE_ABSOLUTE_CLAIM_CONTRADICTION_EXPERIMENT", "false")
    row_off = _normalize(_vg(), claim, statement)

    monkeypatch.setenv("ENABLE_ABSOLUTE_CLAIM_CONTRADICTION_EXPERIMENT", "true")
    row_on = _normalize(_vg(), claim, statement)

    assert str(row_off.get("relevance") or "").upper() == str(row_on.get("relevance") or "").upper()
    assert bool(row_off.get("contradiction_to_absolute_flag", False)) is False
    assert bool(row_on.get("contradiction_to_absolute_flag", False)) is False


def test_population_penalty_is_retained_without_absolute_experiment(monkeypatch):
    monkeypatch.setenv("ENABLE_ABSOLUTE_CLAIM_CONTRADICTION_EXPERIMENT", "false")
    vg = _vg()
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": "In adults with obesity, weight loss may help some joint pain but does not prevent all pain.",
            "relevance": "REFUTES",
            "relevance_score": 0.86,
            "refute_score": 0.84,
            "contradiction_score": 0.82,
            "predicate_match_score": 0.56,
            "subject_match_score": 0.62,
            "object_match_score": 0.54,
            "anchor_match_score": 0.53,
            "population_consistency_score": 0.0,
            "credibility": 0.92,
            "source_url": "https://example.org/trial",
        }
    ]
    _, contradict_mass, _, _, diagnostics = vg._compute_deterministic_masses(
        claim="Losing weight prevents all types of joint pain in adults.",
        evidence_map=evidence_map,
        evidence_count=1,
        source_domains=1,
    )
    row = evidence_map[0]
    assert bool(row.get("population_mismatch_penalty_applied", False))
    assert not bool(row.get("population_mismatch_fatal", False))
    assert bool(row.get("admitted_to_contradict_mass", False))
    assert contradict_mass > 0.0
    assert diagnostics.get("population_policy_mode") == "fatal_or_penalty"


def test_internal_verdict_semantics_unchanged_by_absolute_experiment_flag(monkeypatch):
    claim = "Intervention X improves condition Y."
    payload = {
        "verdict": "UNVERIFIABLE",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN", "evidence_used_ids": []}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Background context with no directional conclusion.",
                "relevance": "NEUTRAL",
                "relevance_score": 0.42,
                "source_url": "https://example.org/context",
            }
        ],
    }

    monkeypatch.setenv("ENABLE_ABSOLUTE_CLAIM_CONTRADICTION_EXPERIMENT", "false")
    out_off = _vg()._enforce_binary_verdict_payload(claim, copy.deepcopy(payload), evidence=[])
    monkeypatch.setenv("ENABLE_ABSOLUTE_CLAIM_CONTRADICTION_EXPERIMENT", "true")
    out_on = _vg()._enforce_binary_verdict_payload(claim, copy.deepcopy(payload), evidence=[])

    assert out_off.get("verdict_internal") == out_on.get("verdict_internal")
    assert out_off.get("verdict") == out_on.get("verdict")


def test_diagnostics_expose_experiment_state(monkeypatch):
    monkeypatch.setenv("ENABLE_ABSOLUTE_CLAIM_CONTRADICTION_EXPERIMENT", "true")
    vg = _vg()
    row = _normalize(
        vg,
        "Hand sanitizer instantly kills 100% of all known germs and viruses.",
        "Most alcohol-based sanitizers are effective against many pathogens but not all.",
    )
    _, _, _, _, diagnostics = vg._compute_deterministic_masses(
        claim="Hand sanitizer instantly kills 100% of all known germs and viruses.",
        evidence_map=[row],
        evidence_count=1,
        source_domains=1,
    )
    assert bool(row.get("absolute_claim_experiment_enabled", False))
    assert "absolute_claim_experiment_path_used" in row
    assert diagnostics.get("absolute_claim_experiment_enabled") is True
    assert diagnostics.get("absolute_claim_experiment_path_used_count", 0) >= 0
