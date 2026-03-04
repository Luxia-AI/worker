from app.services.verdict.verdict_generator import VerdictGenerator


def test_segment_predicate_backoff_prevents_false_unknown():
    vg = VerdictGenerator.__new__(VerdictGenerator)
    # Simulate predicate matcher miss while lexical/anchor support is strong.
    vg.compute_predicate_match = lambda _seg, _stmt: 0.0

    segment = "benefits depend on age and risk profile"
    claim_breakdown = [
        {
            "claim_segment": segment,
            "status": "UNKNOWN",
            "supporting_fact": "",
            "source_url": "",
            "evidence_used_ids": [],
        }
    ]
    evidence_map = [
        {
            "evidence_id": 0,
            "statement": "Screening benefits depend on age and risk profile in adults.",
            "source_url": "https://example.org/evidence",
            "relevance": "SUPPORTS",
            "relevance_score": 0.82,
            "support_strength": 0.72,
            "contradiction_score": 0.08,
            "credibility": 0.95,
            "intervention_match": True,
        }
    ]

    out = vg._align_segments_with_evidence(claim_breakdown, evidence_map, evidence=[])
    assert out[0]["status"] in {"VALID", "PARTIALLY_VALID"}
    assert str(out[0].get("supporting_fact") or "").strip()
