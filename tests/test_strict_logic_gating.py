from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)

    class _TrustPolicyStub:
        def decompose_claim(self, claim):
            return [claim]

        def compute_adaptive_trust(self, claim, evidence, top_k=10):
            return {
                "coverage": 0.8,
                "agreement": 0.85,
                "diversity": 0.6,
                "trust_post": 0.55,
                "is_sufficient": True,
                "num_subclaims": 1,
            }

    vg.trust_policy = _TrustPolicyStub()
    vg._log_subclaim_coverage = lambda *args, **kwargs: None
    return vg


def test_case1_dna_alteration_refutation_forces_false():
    vg = _vg()
    claim = "COVID-19 vaccines alter human DNA"
    evidence = [
        {
            "statement": "mRNA vaccines do not alter human DNA.",
            "source_url": "https://www.cdc.gov/example",
            "final_score": 0.90,
            "credibility": 0.95,
        },
        {
            "statement": "mRNA from vaccines does not integrate into the human genome.",
            "source_url": "https://www.who.int/example",
            "final_score": 0.88,
            "credibility": 0.95,
        },
        {
            "statement": "DNA vectors are used in some vaccine platforms.",
            "source_url": "https://example.org/background",
            "final_score": 0.60,
            "credibility": 0.70,
        },
    ]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.95,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.9},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.88},
            {"evidence_id": 2, "statement": evidence[2]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.6},
        ],
    }
    out = vg._parse_verdict_result(llm_result, claim, evidence)
    assert out["verdict"] == "FALSE"
    assert out["truthfulness_percent"] <= 15.0
    assert out["confidence"] >= 0.75


def test_case2_entity_overlap_only_is_neutral_and_not_valid():
    vg = _vg()
    claim = "COVID-19 vaccines alter human DNA"
    evidence = [
        {
            "statement": "Adenovirus DNA vectors are used in vaccine delivery systems.",
            "source_url": "https://example.org/vector",
            "final_score": 0.85,
            "credibility": 0.9,
        }
    ]
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": evidence[0]["statement"],
            "source_url": evidence[0]["source_url"],
            "relevance": "SUPPORTS",
            "relevance_score": 0.85,
        }
    ]
    normalized = vg._normalize_evidence_map(claim, evidence_map, evidence)
    assert normalized[0]["relevance"] == "NEUTRAL"

    breakdown = [{"claim_segment": claim, "status": "UNKNOWN"}]
    aligned = vg._align_segments_with_evidence(breakdown, normalized, evidence)
    assert aligned[0]["status"] != "VALID"


def test_case3_unverifiable_confidence_is_capped():
    vg = _vg()
    claim = "This claim has insufficient evidence."
    evidence = [
        {
            "statement": "General background context without direct predicate support.",
            "source_url": "https://example.org/bg",
            "final_score": 0.40,
            "credibility": 0.7,
        }
    ]
    llm_result = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.95,
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
    out = vg._parse_verdict_result(llm_result, claim, evidence)
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["confidence"] <= 0.60


def test_case4_mixed_support_with_strong_contradiction_fires_override():
    vg = _vg()
    claim = "mRNA vaccines integrate into human DNA"
    evidence = [
        {
            "statement": "Some hypotheses discuss integration under rare experimental conditions.",
            "source_url": "https://example.org/hypothesis",
            "final_score": 0.62,
            "credibility": 0.6,
        },
        {
            "statement": "mRNA vaccines do not integrate into human DNA.",
            "source_url": "https://www.cdc.gov/refute",
            "final_score": 0.9,
            "credibility": 0.95,
        },
        {
            "statement": "The vaccine mRNA is degraded and does not alter DNA.",
            "source_url": "https://www.nih.gov/refute",
            "final_score": 0.88,
            "credibility": 0.95,
        },
    ]
    llm_result = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "PARTIALLY_VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.62},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.9},
            {"evidence_id": 2, "statement": evidence[2]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.88},
        ],
    }
    out = vg._parse_verdict_result(llm_result, claim, evidence)
    assert out["verdict"] == "FALSE"
    assert out["override_fired"] == "CONTRADICTION_DOMINANCE"
