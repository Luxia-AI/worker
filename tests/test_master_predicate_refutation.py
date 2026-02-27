from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)

    class _TrustPolicyStub:
        def decompose_claim(self, claim):
            return [claim]

        def compute_adaptive_trust(self, claim, evidence, top_k=10):
            return {
                "coverage": 0.88,
                "agreement": 0.9,
                "diversity": 0.62,
                "trust_post": 0.64,
                "is_sufficient": True,
                "num_subclaims": 1,
            }

    vg.trust_policy = _TrustPolicyStub()
    vg._log_subclaim_coverage = lambda *args, **kwargs: None
    vg._last_predicate_queries_generated = []
    return vg


def test_predicate_refute_query_builder_patterns():
    vg = _vg()
    q = vg._predicate_refute_query_hints("mRNA vaccines change your DNA")
    flat = " | ".join(x.lower() for x in q)
    assert "do not" in flat
    assert "cannot" in flat
    assert "no evidence that" in flat
    assert "myth" in flat
    assert "debunked" in flat
    assert "does " in flat


def test_predicate_match_score_tiers():
    vg = _vg()
    exact = vg.compute_predicate_match(
        "mRNA vaccines change human DNA",
        "mRNA vaccines change human DNA",
    )
    close = vg.compute_predicate_match(
        "mRNA vaccines change human DNA",
        "mRNA vaccines alter human DNA",
    )
    none = vg.compute_predicate_match(
        "mRNA vaccines change human DNA",
        "mRNA vaccines are used in vaccination programs",
    )
    assert exact >= 0.99
    assert close >= 0.7
    assert none == 0.0


def test_master_injection_claim_mrna_false_not_unverifiable():
    vg = _vg()
    claim = "mRNA COVID-19 vaccines change your DNA."
    evidence = [
        {
            "statement": "mRNA vaccines do not alter human DNA.",
            "source_url": "https://www.cdc.gov/refute",
            "final_score": 0.92,
            "credibility": 0.95,
        },
        {
            "statement": "mRNA from vaccines does not integrate into the human genome.",
            "source_url": "https://www.who.int/refute",
            "final_score": 0.89,
            "credibility": 0.95,
        },
        {
            "statement": "mRNA vaccines are used to reduce severe COVID-19 outcomes.",
            "source_url": "https://example.org/background",
            "final_score": 0.57,
            "credibility": 0.7,
        },
    ]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.95,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.92},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.89},
            {"evidence_id": 2, "statement": evidence[2]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.57},
        ],
    }
    out = vg._parse_verdict_result(llm_result, claim, evidence)
    assert out["verdict"] == "FALSE"
    assert 5.0 <= out["truthfulness_percent"] <= 15.0
    assert 0.60 <= out["confidence"] <= 0.90
    assert out["analysis_counts"]["map_contradict_signal_max"] > 0.35
    assert any(
        float(ev.get("predicate_match_score", 0.0) or 0.0) >= 0.7 and str(ev.get("relevance") or "") == "REFUTES"
        for ev in (out.get("evidence_map") or [])
    )
    assert out["verdict"] != "UNVERIFIABLE"


def test_calcium_build_claim_maps_to_support_not_neutral():
    vg = _vg()
    claim = "Calcium builds strong bones."
    evidence = [
        {
            "statement": "Calcium is needed to build and maintain strong bones.",
            "source_url": "https://ods.od.nih.gov/factsheets/Calcium-HealthProfessional/",
            "final_score": 0.83,
            "credibility": 0.95,
        }
    ]
    normalized = vg._normalize_evidence_map(
        claim,
        [{"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.7}],
        evidence,
    )
    assert normalized
    assert normalized[0]["relevance"] == "SUPPORTS"
    assert float(normalized[0].get("predicate_match_score", 0.0) or 0.0) >= 0.7


def test_predicate_refute_hints_skip_for_pattern_noun_phrase_claim():
    vg = _vg()
    q = vg._predicate_refute_query_hints("Vitamin D for bone growth in children")
    assert q == []


def test_for_pattern_claim_extracts_support_predicate_and_matches_evidence():
    vg = _vg()
    triplet = vg._extract_canonical_predicate_triplet("Vitamin D for bone growth in children")
    assert triplet["canonical_predicate"] == "support"
    score = vg.compute_predicate_match(
        "Vitamin D for bone growth in children",
        "Vitamin D is necessary for normal bone growth in children.",
    )
    assert score >= 0.7


def test_not_all_quantifier_not_refuted_by_subset_evidence():
    vg = _vg()
    claim = "Not all fats are harmful"
    evidence = [
        {
            "statement": "Trans fats are harmful to health.",
            "source_url": "https://www.bmj.com/content/351/bmj.h3978/rr-8",
            "final_score": 0.82,
            "credibility": 0.95,
        }
    ]
    normalized = vg._normalize_evidence_map(
        claim,
        [{"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "REFUTES", "relevance_score": 0.82}],
        evidence,
    )
    assert normalized
    assert normalized[0]["relevance"] != "REFUTES"

    llm_result = {
        "verdict": "FALSE",
        "confidence": 0.8,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "INVALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "REFUTES", "relevance_score": 0.82}
        ],
    }
    out = vg._parse_verdict_result(llm_result, claim, evidence)
    assert out["verdict"] != "FALSE"
    assert out["claim_breakdown"][0]["status"] != "INVALID"


def test_moderate_claim_not_refuted_by_overload_statement():
    vg = _vg()
    claim = "Moderate coffee consumption does not cause dehydration."
    evidence = [
        {
            "statement": "Caffeine overload is the risk for dehydration, not coffee itself.",
            "source_url": "https://health.clevelandclinic.org/coffee-dehydration",
            "final_score": 0.46,
            "credibility": 0.7,
        }
    ]
    normalized = vg._normalize_evidence_map(
        claim,
        [{"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "REFUTES", "relevance_score": 0.46}],
        evidence,
    )
    assert normalized
    assert normalized[0]["relevance"] != "REFUTES"

    llm_result = {
        "verdict": "FALSE",
        "confidence": 0.75,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "INVALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.46}
        ],
    }
    out = vg._parse_verdict_result(llm_result, claim, evidence)
    assert out["verdict"] != "FALSE"
    assert out["claim_breakdown"][0]["status"] != "INVALID"
