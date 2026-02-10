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

    ranked = await rank_candidates(
        semantic_candidates=[],
        kg_candidates=[],
        query_entities=["vaccines", "autism"],
        query_text="Vaccines do not cause autism.",
        top_k=2,
        round_id="test-round",
        log_manager=None,
    )

    assert len(ranked) == 2
    assert ranked[0]["stance"] == "entails"
    assert ranked[1]["stance"] == "contradicts"
    assert ranked[1]["final_score"] < ranked[0]["final_score"]
