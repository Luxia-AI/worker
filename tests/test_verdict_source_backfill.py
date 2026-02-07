from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy
from app.services.verdict.verdict_generator import VerdictGenerator


def test_claim_breakdown_backfills_source_url():
    vg = object.__new__(VerdictGenerator)
    vg.trust_policy = AdaptiveTrustPolicy()

    claim = "Vitamin D supports bone health."
    evidence = [
        {
            "statement": "Vitamin D supports bone health and calcium metabolism.",
            "source_url": "https://www.nih.gov/health-information",
            "final_score": 0.88,
            "sem_score": 0.82,
        }
    ]
    llm_result = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.5,
        "truthfulness_percent": 50.0,
        "rationale": "test",
        "claim_breakdown": [
            {
                "claim_segment": "Vitamin D supports bone health.",
                "status": "VALID",
                "supporting_fact": "Vitamin D supports bone health and calcium metabolism.",
                "source_url": "",
            }
        ],
        "evidence_map": [],
        "key_findings": [],
    }

    parsed = vg._parse_verdict_result(llm_result, claim, evidence)
    segment = parsed["claim_breakdown"][0]

    assert segment["source_url"] == "https://www.nih.gov/health-information"
    assert segment["evidence_used_ids"] == [0]
