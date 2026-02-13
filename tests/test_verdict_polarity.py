from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)

    class _TrustPolicyStub:
        def decompose_claim(self, claim):
            return [claim]

        def compute_adaptive_trust(self, claim, evidence, top_k=10):
            return {
                "coverage": 0.0,
                "agreement": 1.0,
                "diversity": 0.5,
                "trust_post": 0.0,
                "is_sufficient": True,
                "num_subclaims": 1,
            }

    vg.trust_policy = _TrustPolicyStub()
    vg._log_subclaim_coverage = lambda *args, **kwargs: None
    return vg


def test_negative_segment_negative_evidence_entails_and_reconciles_true():
    vg = _vg()
    claim = "Antibiotics do not work against viruses."
    evidence = [
        {
            "statement": "Antibiotics do not treat viral infections.",
            "source_url": "https://www.cdc.gov/example",
            "final_score": 0.9,
            "credibility": 0.95,
        }
    ]
    llm_result = {
        "verdict": "FALSE",
        "confidence": 0.8,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Antibiotics do not work against viruses",
                "status": "VALID",
                "supporting_fact": "Antibiotics do not treat viral infections.",
                "source_url": "https://www.cdc.gov/example",
            }
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    polarity = vg._segment_polarity(
        "Antibiotics do not work against viruses",
        "Antibiotics do not treat viral infections.",
        stance="neutral",
    )
    parsed = vg._parse_verdict_result(llm_result, claim, evidence)

    assert polarity == "entails"
    assert parsed["verdict"] == "UNVERIFIABLE"
    assert 35.0 <= parsed["truth_score_percent"] <= 65.0
    assert "evidence_quality_percent" in parsed


def test_positive_segment_negative_evidence_is_contradiction_and_false():
    vg = _vg()
    claim = "Antibiotics work against viruses."

    polarity = vg._segment_polarity(
        "Antibiotics work against viruses",
        "Antibiotics do not treat viral infections.",
        stance="neutral",
    )
    reconciled = vg._reconcile_verdict_with_breakdown(
        claim,
        [
            {
                "claim_segment": "Antibiotics work against viruses",
                "status": "INVALID",
                "supporting_fact": "Antibiotics do not treat viral infections.",
                "source_url": "https://www.cdc.gov/example",
            }
        ],
    )

    assert polarity == "contradicts"
    assert reconciled["verdict"] == "FALSE"


def test_ineffective_and_do_not_work_are_symmetric_entailment():
    vg = _vg()
    polarity = vg._segment_polarity(
        "Antibiotics are ineffective against viruses",
        "Antibiotics do not work against viruses.",
        stance="neutral",
    )
    assert polarity == "entails"


def test_antibiotic_viral_misuse_statement_entails_negative_efficacy_claim():
    vg = _vg()
    polarity = vg._segment_polarity(
        "Antibiotics do not work against viruses like the common cold.",
        "Antibiotics are often misused in treating viral infections like the common cold.",
        stance="supports",
    )
    assert polarity == "entails"


def test_hypertension_symptom_claim_is_contradicted_by_silent_condition_statement():
    vg = _vg()
    polarity = vg._segment_polarity(
        "High blood pressure usually has noticeable symptoms like headaches.",
        "High blood pressure is often a silent condition and usually has no symptoms.",
        stance="neutral",
    )
    assert polarity == "contradicts"


def test_hypertension_symptom_predicate_guard_rejects_treatment_only_statement():
    assert not VerdictGenerator._predicate_guard_ok(
        "High blood pressure usually has noticeable symptoms like headaches.",
        "Management of high blood pressure focuses on blood pressure control using medications.",
    )


def test_smoking_stroke_segment_rejects_heart_only_statement():
    assert not VerdictGenerator._predicate_guard_ok(
        "Smoking increases the risk of stroke.",
        "Smoking increases the risk of heart disease.",
    )


def test_diabetes_lifestyle_segment_rejects_etiology_only_statement():
    assert not VerdictGenerator._predicate_guard_ok(
        "Type 2 diabetes can sometimes be managed with lifestyle changes.",
        "Type 2 diabetes is caused by insulin resistance.",
    )


def test_diabetes_medication_reduction_segment_accepts_supervised_statement():
    assert VerdictGenerator._predicate_guard_ok(
        "Some people can reduce or stop diabetes medications under medical supervision.",
        "Some people with type 2 diabetes can reduce diabetes medications under medical "
        "supervision after lifestyle changes.",
    )


def test_rationale_rewrite_removes_partially_true_for_true_verdict():
    vg = _vg()
    reconciled = {"verdict": "TRUE", "unresolved_segments": 0}
    breakdown = [
        {
            "claim_segment": "Vaccines do not cause autism",
            "status": "VALID",
            "supporting_fact": "Vaccines do not cause autism.",
            "source_url": "https://example.org",
        }
    ]
    out = vg._rewrite_rationale_from_breakdown(
        "The claim is partially true based on evidence.",
        breakdown,
        reconciled,
    )
    assert "partially true" not in out.lower()


def test_unknown_segment_without_fact_is_aligned_and_marked_invalid_for_negation_mismatch():
    vg = _vg()
    claim = "Handwashing does not reduce the spread of infectious diseases."
    evidence = [
        {
            "statement": "Good handwashing prevents the spread of germs.",
            "source_url": "https://www.esneft.nhs.uk/service/infection-control/",
            "semantic_score": 1.0,
            "final_score": 0.85,
            "credibility": 0.95,
            "stance": "supports",
        }
    ]
    llm_result = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.35,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Handwashing does not reduce the spread of infectious diseases",
                "status": "UNKNOWN",
                "supporting_fact": "",
                "source_url": "",
            }
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)

    assert parsed["verdict"] == "FALSE"
    assert parsed["claim_breakdown"][0]["status"] == "INVALID"


def test_strong_therapeutic_profile_triggers_misleading_for_weak_only_support():
    vg = _vg()
    claim = "Drinking green tea cures cancer."
    evidence = [
        {
            "statement": "Green tea cures cancer in anecdotal reports.",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "semantic_score": 0.60,
            "final_score": 0.60,
            "credibility": 0.90,
            "stance": "entails",
        }
    ]
    llm_result = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.35,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Drinking green tea cures cancer",
                "status": "UNKNOWN",
                "supporting_fact": "",
                "source_url": "",
            }
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)
    assert parsed["verdict"] == "UNVERIFIABLE"
