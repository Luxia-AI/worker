from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)
    vg._last_predicate_queries_generated = []
    return vg


def _run_refute_admission(claim: str, item: dict):
    vg = _vg()
    evidence_map = [dict(item)]
    support_mass, contradict_mass, neutral_mass, sufficiency, diagnostics = vg._compute_deterministic_masses(
        claim=claim,
        evidence_map=evidence_map,
        evidence_count=len(evidence_map),
        source_domains=1,
    )
    return evidence_map[0], support_mass, contradict_mass, neutral_mass, sufficiency, diagnostics


def test_clear_negation_contradiction_is_admitted_to_contradict_mass():
    row, _, contradict_mass, _, _, diagnostics = _run_refute_admission(
        "Omega-3 supplements do not reduce inflammation.",
        {
            "evidence_id": 0,
            "statement": "Omega-3 supplements reduce inflammation markers in trials.",
            "relevance": "REFUTES",
            "relevance_score": 0.88,
            "refute_score": 0.82,
            "contradiction_score": 0.84,
            "predicate_match_score": 0.61,
            "subject_match_score": 0.58,
            "object_match_score": 0.53,
            "anchor_match_score": 0.55,
            "credibility": 0.91,
            "source_url": "https://example.org/nih",
        },
    )
    assert contradict_mass > 0.0
    assert diagnostics["refute_admitted_count"] > 0
    assert bool(row.get("admitted_to_contradict_mass", False))
    assert row.get("final_stance") == "REFUTES"


def test_absolute_claim_contradiction_patterns_are_admitted():
    claims = [
        "Vitamin X always prevents colds.",
        "Supplement Y never causes side effects.",
        "Drug Z is the only effective treatment for migraine.",
        "Herb Q is 100% guaranteed to cure insomnia.",
    ]
    for claim in claims:
        row, _, contradict_mass, _, _, _ = _run_refute_admission(
            claim,
            {
                "evidence_id": 0,
                "statement": "Evidence shows mixed outcomes and documented exceptions.",
                "relevance": "REFUTES",
                "relevance_score": 0.70,
                "refute_score": 0.57,
                "contradiction_score": 0.58,
                "predicate_match_score": 0.14,
                "subject_match_score": 0.23,
                "object_match_score": 0.09,
                "anchor_match_score": 0.40,
                "credibility": 0.82,
                "source_url": "https://example.org/review",
            },
        )
        assert contradict_mass > 0.0
        assert bool(row.get("admitted_to_contradict_mass", False))


def test_comparator_contradiction_is_admitted():
    row, _, contradict_mass, _, _, _ = _run_refute_admission(
        "Treatment A is more effective than Treatment B.",
        {
            "evidence_id": 0,
            "statement": "High-quality trials found A is not more effective than B.",
            "relevance": "REFUTES",
            "relevance_score": 0.86,
            "refute_score": 0.79,
            "contradiction_score": 0.81,
            "predicate_match_score": 0.62,
            "subject_match_score": 0.61,
            "object_match_score": 0.54,
            "anchor_match_score": 0.52,
            "comparator_consistency_score": 1.0,
            "credibility": 0.90,
            "source_url": "https://example.org/meta",
        },
    )
    assert contradict_mass > 0.0
    assert bool(row.get("admitted_to_contradict_mass", False))


def test_topical_wrong_subject_is_blocked():
    row, _, contradict_mass, _, _, diagnostics = _run_refute_admission(
        "Fish oil reduces blood pressure.",
        {
            "evidence_id": 0,
            "statement": "An unrelated intervention did not improve outcomes.",
            "relevance": "REFUTES",
            "relevance_score": 0.90,
            "refute_score": 0.88,
            "contradiction_score": 0.85,
            "predicate_match_score": 0.44,
            "subject_match_score": 0.01,
            "object_match_score": 0.01,
            "anchor_match_score": 0.04,
            "credibility": 0.88,
            "source_url": "https://example.org/unrelated",
        },
    )
    assert contradict_mass == 0.0
    assert diagnostics["refute_labeled_count"] > 0
    assert diagnostics["refute_admitted_count"] == 0
    assert str(row.get("refute_block_reason") or "").strip() in {
        "subject_object_mismatch",
        "low_subject_object_alignment",
    }


def test_topical_wrong_object_is_blocked():
    row, _, contradict_mass, _, _, diagnostics = _run_refute_admission(
        "Magnesium supplements reduce migraine frequency.",
        {
            "evidence_id": 0,
            "statement": "Magnesium has no effect on skin hydration outcomes.",
            "relevance": "REFUTES",
            "relevance_score": 0.84,
            "refute_score": 0.80,
            "contradiction_score": 0.79,
            "predicate_match_score": 0.52,
            "subject_match_score": 0.68,
            "object_match_score": 0.01,
            "anchor_match_score": 0.30,
            "credibility": 0.87,
            "source_url": "https://example.org/offtarget",
        },
    )
    assert contradict_mass == 0.0
    assert diagnostics["refute_labeled_count"] > 0
    assert diagnostics["refute_admitted_count"] == 0
    assert str(row.get("refute_block_reason") or "").strip() in {
        "low_object_semantic_match",
        "low_subject_object_alignment",
    }


def test_timeframe_population_mismatch_blocks_refute():
    row, _, contradict_mass, _, _, diagnostics = _run_refute_admission(
        "Intervention X reduces stroke risk in adults over 65 within 12 months.",
        {
            "evidence_id": 0,
            "statement": "Contradictory signal from a pediatric short-term cohort.",
            "relevance": "REFUTES",
            "relevance_score": 0.81,
            "refute_score": 0.76,
            "contradiction_score": 0.74,
            "predicate_match_score": 0.48,
            "subject_match_score": 0.55,
            "object_match_score": 0.46,
            "anchor_match_score": 0.47,
            "timeframe_consistency_score": 0.0,
            "population_consistency_score": 0.0,
            "credibility": 0.90,
            "source_url": "https://example.org/mismatch",
        },
    )
    assert contradict_mass == 0.0
    assert diagnostics["refute_admitted_count"] == 0
    assert str(row.get("refute_block_reason") or "").strip() in {
        "timeframe_mismatch",
        "population_mismatch",
        "population_mismatch_fatal",
    }


def test_admitted_refute_is_not_silently_dropped_before_mass_aggregation():
    row, _, contradict_mass, _, _, diagnostics = _run_refute_admission(
        "Claim X is always safe.",
        {
            "evidence_id": 0,
            "statement": "Evidence documents adverse events and contraindications.",
            "relevance": "REFUTES",
            "relevance_score": 0.83,
            "refute_score": 0.78,
            "contradiction_score": 0.80,
            "predicate_match_score": 0.56,
            "subject_match_score": 0.52,
            "object_match_score": 0.49,
            "anchor_match_score": 0.51,
            "credibility": 0.89,
            "source_url": "https://example.org/guideline",
        },
    )
    assert bool(row.get("admitted_to_contradict_mass", False))
    assert contradict_mass > 0.0
    assert diagnostics["refute_admitted_count"] == 1
    assert diagnostics["refute_zero_with_labels"] is False


def test_refute_candidate_is_not_neutralized_before_admission_review():
    vg = _vg()
    claim = "Herbal remedy X cures hypertension."
    statement = "There is no evidence that herbal remedy X cures hypertension."
    normalized = vg._normalize_evidence_map(
        claim=claim,
        evidence_map=[
            {
                "evidence_id": 0,
                "statement": statement,
                "relevance": "REFUTES",
                "relevance_score": 0.68,
                "source_url": "https://example.org/review",
            }
        ],
        evidence=[
            {
                "statement": statement,
                "source_url": "https://example.org/review",
                "final_score": 0.68,
                "credibility": 0.86,
            }
        ],
    )
    assert normalized
    row = normalized[0]
    assert str(row.get("relevance") or "").upper() == "REFUTES"
    if not bool(row.get("refute_gate_passed", False)):
        assert str(row.get("refute_gate_block_reason") or "").strip()


def test_absolute_claim_partial_effect_forms_refute_candidate_and_is_admitted():
    vg = _vg()
    claim = "Antibiotics always cure bacterial infections."
    statement = "Antibiotics treat bacterial infections in many cases but not all."
    normalized = vg._normalize_evidence_map(
        claim=claim,
        evidence_map=[
            {
                "evidence_id": 0,
                "statement": statement,
                "relevance": "NEUTRAL",
                "relevance_score": 0.70,
                "source_url": "https://example.org/review",
            }
        ],
        evidence=[
            {
                "statement": statement,
                "source_url": "https://example.org/review",
                "final_score": 0.70,
                "credibility": 0.87,
            }
        ],
    )
    assert normalized
    row = normalized[0]
    assert bool(row.get("claim_absolute_flag", False))
    assert bool(row.get("contradiction_to_absolute_flag", False))
    assert bool(row.get("refute_candidate", False))
    assert "absolute_claim_non_universal_contradiction" in str(row.get("refute_candidate_reason") or "")

    row2, _, contradict_mass, _, _, diagnostics = _run_refute_admission(claim, row)
    assert bool(row2.get("admitted_to_contradict_mass", False))
    assert contradict_mass > 0.0
    assert diagnostics["refute_admitted_count"] > 0


def test_completely_prevents_claim_contradicted_by_incomplete_protection():
    row, _, contradict_mass, _, _, _ = _run_refute_admission(
        "Vitamin C completely prevents the common cold.",
        {
            "evidence_id": 0,
            "statement": "Vitamin C may reduce duration and severity, but does not fully prevent colds.",
            "relevance": "REFUTES",
            "relevance_score": 0.84,
            "refute_score": 0.83,
            "contradiction_score": 0.80,
            "predicate_match_score": 0.52,
            "subject_match_score": 0.62,
            "object_match_score": 0.58,
            "anchor_match_score": 0.55,
            "credibility": 0.90,
            "source_url": "https://example.org/meta",
        },
    )
    assert bool(row.get("contradiction_to_absolute_flag", False))
    assert bool(row.get("partial_effect_contradiction_flag", False))
    assert bool(row.get("admitted_to_contradict_mass", False))
    assert contradict_mass > 0.0


def test_hundred_percent_claim_contradicted_by_less_than_total_effect():
    row, _, contradict_mass, _, _, _ = _run_refute_admission(
        "Hand sanitizer instantly kills 100% of all known germs and viruses.",
        {
            "evidence_id": 0,
            "statement": "Most alcohol-based sanitizers are effective against many pathogens but not all.",
            "relevance": "REFUTES",
            "relevance_score": 0.82,
            "refute_score": 0.80,
            "contradiction_score": 0.78,
            "predicate_match_score": 0.40,
            "subject_match_score": 0.55,
            "object_match_score": 0.52,
            "anchor_match_score": 0.50,
            "credibility": 0.88,
            "source_url": "https://example.org/guideline",
        },
    )
    assert bool(row.get("universal_scope_contradiction_flag", False))
    assert bool(row.get("admitted_to_contradict_mass", False))
    assert contradict_mass > 0.0


def test_only_effective_claim_contradicted_by_multiple_options():
    row, _, contradict_mass, _, _, _ = _run_refute_admission(
        "Drug A is the only effective treatment for condition Z.",
        {
            "evidence_id": 0,
            "statement": "Drug A is one of several effective treatment options for condition Z.",
            "relevance": "REFUTES",
            "relevance_score": 0.83,
            "refute_score": 0.81,
            "contradiction_score": 0.79,
            "predicate_match_score": 0.48,
            "subject_match_score": 0.57,
            "object_match_score": 0.59,
            "anchor_match_score": 0.51,
            "credibility": 0.89,
            "source_url": "https://example.org/recommendation",
        },
    )
    assert bool(row.get("exclusivity_contradiction_flag", False))
    assert bool(row.get("admitted_to_contradict_mass", False))
    assert contradict_mass > 0.0


def test_broad_adult_claim_uses_population_penalty_not_fatal_block():
    row, _, contradict_mass, _, _, diagnostics = _run_refute_admission(
        "Losing weight prevents all types of joint pain in adults.",
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
        },
    )
    assert bool(row.get("population_mismatch_penalty_applied", False))
    assert not bool(row.get("population_mismatch_fatal", False))
    assert bool(row.get("admitted_to_contradict_mass", False))
    assert contradict_mass > 0.0
    assert diagnostics["refute_admitted_count"] > 0


def test_truly_incompatible_population_remains_fatal_block():
    row, _, contradict_mass, _, _, diagnostics = _run_refute_admission(
        "Intervention X prevents condition Y in adult men.",
        {
            "evidence_id": 0,
            "statement": "In pediatric girls, intervention X had mixed outcomes and did not prevent condition Y.",
            "relevance": "REFUTES",
            "relevance_score": 0.85,
            "refute_score": 0.83,
            "contradiction_score": 0.80,
            "predicate_match_score": 0.50,
            "subject_match_score": 0.58,
            "object_match_score": 0.52,
            "anchor_match_score": 0.49,
            "population_consistency_score": 0.0,
            "credibility": 0.91,
            "source_url": "https://example.org/peds",
        },
    )
    assert bool(row.get("population_mismatch_fatal", False))
    assert str(row.get("refute_block_reason") or "").strip() == "population_mismatch_fatal"
    assert not bool(row.get("admitted_to_contradict_mass", False))
    assert contradict_mass == 0.0
    assert diagnostics["refute_admitted_count"] == 0
