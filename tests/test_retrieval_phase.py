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
