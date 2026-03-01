from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.corrective.pipeline.retrieval_phase import retrieve_candidates


@pytest.mark.asyncio
async def test_retrieve_candidates_no_topics_uses_fallback_paths():
    vdb_retriever = MagicMock()
    vdb_retriever.search = AsyncMock(
        return_value=[
            {
                "statement": "Liver and kidneys naturally remove waste products.",
                "score": 0.72,
                "source_url": "https://nih.gov",
            }
        ]
    )
    vdb_retriever.fetch_by_ids = MagicMock(return_value=[])

    kg_retriever = MagicMock()
    kg_retriever.retrieve = AsyncMock(return_value=[])

    lexical_index = MagicMock()
    lexical_index.search = MagicMock(return_value=[{"fact_id": "fact-1", "bm25": 0.12}])

    dedup_sem, kg_candidates, metrics = await retrieve_candidates(
        vdb_retriever=vdb_retriever,
        kg_retriever=kg_retriever,
        queries=["body cleanse liver kidney evidence"],
        all_entities=["liver", "kidney"],
        top_k=5,
        round_id="test-round",
        topics=[],
        lexical_index=lexical_index,
        include_metrics=True,
    )

    assert dedup_sem
    assert kg_candidates == []
    assert metrics["sem_filtered"] >= 1
    assert vdb_retriever.search.await_count == 1
    assert lexical_index.search.call_count == 1


@pytest.mark.asyncio
async def test_retrieve_candidates_triggers_kg_fallback_when_initial_kg_is_empty():
    vdb_retriever = MagicMock()
    vdb_retriever.search = AsyncMock(return_value=[])
    vdb_retriever.fetch_by_ids = MagicMock(return_value=[])

    kg_retriever = MagicMock()
    kg_retriever.retrieve = AsyncMock(
        side_effect=[
            [],
            [
                {
                    "subject": "omega-3",
                    "relation": "reduces",
                    "object": "inflammation",
                    "score": 0.62,
                    "source_url": "https://example.org/e1",
                }
            ],
        ]
    )

    dedup_sem, kg_candidates, metrics = await retrieve_candidates(
        vdb_retriever=vdb_retriever,
        kg_retriever=kg_retriever,
        queries=["omega-3 inflammation evidence"],
        all_entities=[],
        top_k=5,
        round_id="test-kg-fallback",
        topics=[],
        lexical_index=None,
        query_text="Omega-3 supplementation may reduce inflammation",
        include_metrics=True,
    )

    assert dedup_sem == []
    assert len(kg_candidates) == 1
    assert kg_retriever.retrieve.await_count == 2
    assert metrics["kg_fallback_triggered"] == 1
    assert metrics["kg_zero_signal"] == 0
