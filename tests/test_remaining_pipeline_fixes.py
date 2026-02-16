from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)

    class _TrustPolicyStub:
        def decompose_claim(self, claim):
            return [claim]

        def compute_adaptive_trust(self, claim, evidence, top_k=10):
            return {
                "coverage": 0.85,
                "agreement": 0.9,
                "diversity": 0.6,
                "trust_post": 0.6,
                "is_sufficient": True,
                "num_subclaims": 1,
            }

    vg.trust_policy = _TrustPolicyStub()
    vg._log_subclaim_coverage = lambda *args, **kwargs: None
    vg._last_predicate_queries_generated = []
    return vg


def test_mrna_change_dna_forced_false_with_low_truthfulness():
    vg = _vg()
    claim = "mRNA COVID-19 vaccines change human DNA"
    evidence = [
        {
            "statement": "mRNA vaccines do not alter human DNA.",
            "source_url": "https://www.cdc.gov/refute",
            "final_score": 0.91,
            "credibility": 0.95,
        },
        {
            "statement": "Vaccine mRNA does not integrate into the human genome.",
            "source_url": "https://www.who.int/refute",
            "final_score": 0.88,
            "credibility": 0.95,
        },
        {
            "statement": "mRNA vaccines are used to prevent severe COVID-19.",
            "source_url": "https://example.org/background",
            "final_score": 0.56,
            "credibility": 0.7,
        },
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.95,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.91},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.88},
            {"evidence_id": 2, "statement": evidence[2]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.56},
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["verdict"] == "FALSE"
    assert out["truthfulness_percent"] <= 15.0
    assert out["confidence"] >= 0.75
    assert out["contradiction_override_fired"] is True


def test_background_only_evidence_stays_unknown_and_truthfulness_bounded():
    vg = _vg()
    claim = "X cures Y"
    evidence = [
        {
            "statement": "X is a commonly discussed intervention.",
            "source_url": "https://example.org/bg",
            "final_score": 0.7,
            "credibility": 0.7,
        }
    ]
    llm = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "PARTIALLY_VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.7}
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] == "UNKNOWN"
    assert out["truthfulness_percent"] <= 45.0
    assert out["truthfulness_invariant_applied"] is True


def test_mixed_support_and_explicit_refute_refute_wins():
    vg = _vg()
    claim = "Vitamin C always prevents the common cold"
    evidence = [
        {
            "statement": "Vitamin C may slightly reduce cold duration in some groups.",
            "source_url": "https://example.org/support",
            "final_score": 0.72,
            "credibility": 0.8,
        },
        {
            "statement": "Vitamin C does not prevent the common cold in the general population.",
            "source_url": "https://www.nih.gov/refute",
            "final_score": 0.89,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.92,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.72},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.89},
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["explicit_refutes_found"] is True
    assert out["claim_breakdown"][0]["status"] in {"INVALID", "PARTIALLY_INVALID"}


def test_unverifiable_confidence_cap_is_enforced():
    vg = _vg()
    claim = "Claim with no decisive evidence."
    evidence = [
        {
            "statement": "Background context only.",
            "source_url": "https://example.org",
            "final_score": 0.4,
            "credibility": 0.6,
        }
    ]
    llm = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.97,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": evidence[0]["statement"],
                "relevance": "NEUTRAL",
                "relevance_score": 0.4,
            }
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["confidence"] <= 0.60


def test_predicate_targeted_query_generation_patterns():
    vg = _vg()
    claim = "mRNA vaccines change your DNA"
    queries = vg._predicate_refute_query_hints(claim)
    text = " | ".join(q.lower() for q in queries)
    assert "do not" in text
    assert "cannot" in text
    assert "no evidence" in text
    assert "does not" in text


def test_valid_support_segment_not_downgraded_to_unverifiable():
    vg = _vg()
    claim = "Calcium builds strong bones"
    evidence = [
        {
            "statement": "Calcium is needed to build and maintain strong bones.",
            "source_url": "https://ods.od.nih.gov/factsheets/Calcium-Consumer/",
            "final_score": 0.58,
            "sem_score": 0.88,
            "credibility": 0.95,
        },
        {
            "statement": "Calcium is stored in bones and teeth.",
            "source_url": "https://ods.od.nih.gov/factsheets/Calcium-Consumer/",
            "final_score": 0.66,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.78,
        "rationale": "Claim is partially supported but phrasing is somewhat ambiguous.",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.73},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.36},
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE"}


def test_food_claim_not_refuted_by_supplement_only_evidence():
    vg = _vg()
    claim = "A healthy diet rich in vegetables and fruit may help reduce the risk of some cancers"
    evidence = [
        {
            "statement": (
                "There is no clear evidence that vitamin A supplementation decreases cancer risk "
                "in people consuming a healthy diet."
            ),
            "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK13273/",
            "final_score": 0.62,
            "credibility": 0.95,
        },
        {
            "statement": "Fruit and vegetables can help reduce risk of some cancers.",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/article/abc",
            "final_score": 0.71,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "FALSE",
        "confidence": 0.92,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "REFUTES", "relevance_score": 0.8},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.8},
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    ev_map = out["evidence_map"]
    supplement_row = next(x for x in ev_map if "supplementation" in x["statement"].lower())
    assert supplement_row["intervention_match"] is False
    assert supplement_row["relevance"] != "REFUTES"
    assert out["verdict"] != "FALSE"


def test_vitamin_c_immune_function_paraphrase_is_not_unknown():
    vg = _vg()
    claim = "Vitamin C contributes to the normal function of the immune system"
    evidence = [
        {
            "statement": "Vitamin C helps your immune system.",
            "source_url": "https://health.clevelandclinic.org/vitamin-c",
            "final_score": 0.60,
            "sem_score": 1.0,
            "credibility": 0.95,
        },
        {
            "statement": "Vitamin C supports immune health.",
            "source_url": "https://health.clevelandclinic.org/vitamin-c",
            "final_score": 0.50,
            "sem_score": 0.99,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.6,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.60},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.50},
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    segment = out["claim_breakdown"][0]
    assert segment["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE"}


def test_vitamin_c_immune_function_paraphrase_not_unverifiable():
    vg = _vg()
    claim = "Vitamin C contributes to the normal function of the immune system"
    evidence = [
        {
            "statement": "Vitamin C helps your immune system.",
            "source_url": "https://health.clevelandclinic.org/vitamin-c",
            "final_score": 0.60,
            "sem_score": 1.0,
            "credibility": 0.95,
        },
        {
            "statement": "Vitamin C supports immune health.",
            "source_url": "https://health.clevelandclinic.org/vitamin-c",
            "final_score": 0.50,
            "sem_score": 0.99,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.6,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.6},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.5},
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE"}
