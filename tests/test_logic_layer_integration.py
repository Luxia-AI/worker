import json
from pathlib import Path

from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)

    class _TrustPolicyStub:
        def decompose_claim(self, claim):
            return [claim]

        def compute_adaptive_trust(self, claim, evidence, top_k=10):
            return {
                "coverage": 0.7,
                "agreement": 0.6,
                "diversity": 0.6,
                "trust_post": 0.55,
                "is_sufficient": True,
                "num_subclaims": 2,
                "strong_covered": 1,
                "contradicted_subclaims": 1,
            }

    vg.trust_policy = _TrustPolicyStub()
    vg._log_subclaim_coverage = lambda *args, **kwargs: None
    return vg


def test_hedged_support_with_diverse_refutation_never_returns_true():
    fixture_path = Path(__file__).parent / "fixtures" / "logic_layer_cases.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    case = data["hedged_support_vs_diverse_refute"]

    claim = case["claim"]
    evidence = case["evidence"]
    llm_result = {
        "verdict": "TRUE",
        "confidence": 0.93,
        "rationale": "Model says true.",
        "claim_breakdown": [
            {
                "claim_segment": claim,
                "status": "INVALID",
                "supporting_fact": evidence[1]["statement"],
                "source_url": evidence[1]["source_url"],
            },
            {
                "claim_segment": "severe disease in all adults",
                "status": "PARTIALLY_INVALID",
                "supporting_fact": evidence[2]["statement"],
                "source_url": evidence[2]["source_url"],
            },
        ],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": evidence[0]["statement"],
                "relevance": "SUPPORTS",
                "relevance_score": 0.42,
                "source_url": evidence[0]["source_url"],
            },
            {
                "evidence_id": 1,
                "statement": evidence[1]["statement"],
                "relevance": "CONTRADICTS",
                "relevance_score": 0.82,
                "source_url": evidence[1]["source_url"],
            },
            {
                "evidence_id": 2,
                "statement": evidence[2]["statement"],
                "relevance": "CONTRADICTS",
                "relevance_score": 0.88,
                "source_url": evidence[2]["source_url"],
            },
            {
                "evidence_id": 3,
                "statement": evidence[3]["statement"],
                "relevance": "CONTRADICTS",
                "relevance_score": 0.79,
                "source_url": evidence[3]["source_url"],
            },
        ],
        "key_findings": [],
    }

    parsed = _vg()._parse_verdict_result(llm_result, claim, evidence)

    assert parsed["verdict"] != "TRUE"
    assert "strictness_profile" in parsed
    assert "override_fired" in parsed
    if parsed["verdict"] == "FALSE":
        assert float(parsed["confidence"]) >= 0.55
