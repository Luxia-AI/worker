import pytest

from app.services.corrective.pipeline.ranking_phase import rank_candidates


@pytest.mark.asyncio
async def test_ranking_phase_penalizes_and_deprioritizes_contradicting_evidence(monkeypatch):
    async_ranked = [
        {
            "statement": "Infant vaccines may cause autism.",
            "source_url": "https://example.com/contra",
            "final_score": 0.92,
            "sem_score": 0.92,
            "credibility": 0.8,
            "entity_overlap": 0.9,
            "recency": 0.8,
        },
        {
            "statement": "Vaccines do not cause autism.",
            "source_url": "https://example.com/support",
            "final_score": 0.78,
            "sem_score": 0.78,
            "credibility": 0.9,
            "entity_overlap": 0.9,
            "recency": 0.8,
        },
    ]

    def _fake_hybrid_rank(*args, **kwargs):
        return [dict(item) for item in async_ranked]

    monkeypatch.setattr("app.services.corrective.pipeline.ranking_phase.hybrid_rank", _fake_hybrid_rank)
    monkeypatch.setattr(
        "app.services.corrective.pipeline.ranking_phase._ENTAILMENT_VERIFIER",
        type("V", (), {"verify_refutes": lambda self, claim, candidates, top_n=5: {}})(),
    )

    ranked = await rank_candidates(
        semantic_candidates=[],
        kg_candidates=[],
        query_entities=["vaccines", "autism"],
        query_text="Vaccines do not cause autism.",
        top_k=2,
        round_id="test-round",
        log_manager=None,
    )

    # Contradicting evidence should remain admitted but bounded, while
    # non-contradicting evidence still reaches top-k.
    assert len(ranked) == 2
    assert any(item["stance"] == "entails" for item in ranked)
    assert sum(1 for item in ranked if item["stance"] == "contradicts") <= 1


@pytest.mark.asyncio
async def test_ranking_phase_stage2_verifier_receives_only_stage1_candidates(monkeypatch):
    async_ranked = [
        {
            "statement": "Dietary supplements require FDA pre-market effectiveness approval.",
            "source_url": "https://example.org/a",
            "final_score": 0.80,
            "sem_score": 0.10,
            "kg_score": 0.90,
            "claim_overlap": 0.30,
            "contradict_score": 0.05,
            "negation_anchor_overlap": 0.0,
            "predicate_match_score": 0.80,
            "credibility": 0.9,
            "support_score": 0.7,
        },
        {
            "statement": "FDA does not determine dietary supplement effectiveness before marketing.",
            "source_url": "https://example.org/b",
            "final_score": 0.78,
            "sem_score": 0.12,
            "kg_score": 0.88,
            "claim_overlap": 0.32,
            "contradict_score": 0.85,
            "negation_anchor_overlap": 0.8,
            "predicate_match_score": 0.35,
            "credibility": 0.9,
            "support_score": 0.2,
        },
    ]

    def _fake_hybrid_rank(*args, **kwargs):
        return [dict(item) for item in async_ranked]

    seen = {"count": 0}

    class _Verifier:
        def verify_refutes(self, claim, candidates, top_n=5):
            seen["count"] = len(candidates)
            return {}

    monkeypatch.setattr("app.services.corrective.pipeline.ranking_phase.hybrid_rank", _fake_hybrid_rank)
    monkeypatch.setattr("app.services.corrective.pipeline.ranking_phase._ENTAILMENT_VERIFIER", _Verifier())

    await rank_candidates(
        semantic_candidates=[],
        kg_candidates=[],
        query_entities=["fda", "dietary supplements"],
        query_text="FDA has approved all dietary supplements for effectiveness before they are sold.",
        top_k=2,
        round_id="test-stage2-candidates",
        log_manager=None,
    )

    # Only the contradiction-like candidate should pass stage1 to stage2.
    assert seen["count"] == 1
