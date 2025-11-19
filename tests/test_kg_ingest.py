from unittest.mock import AsyncMock, patch

import pytest

from app.services.kg.kg_ingest import KGIngest


@pytest.mark.asyncio
async def test_kg_ingest_success():
    ingest = KGIngest()

    triples = [
        {
            "subject": "vitamin d",
            "relation": "reduces",
            "object": "fracture",
            "confidence": 0.91,
            "source_url": "https://who.int/article",
            "published_at": "2024-01-01T00:00:00Z",
        }
    ]

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=None)

    # Create a proper async context manager mock
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_session
    mock_context_manager.__aexit__.return_value = None

    with patch.object(ingest.client, "session", return_value=mock_context_manager):
        count = await ingest.ingest_triples(triples)
        assert count == 1
        mock_session.run.assert_called_once()


@pytest.mark.asyncio
async def test_kg_ingest_skips_bad_triple():
    ingest = KGIngest()

    triples = [{"subject": "", "relation": "has", "object": "x"}]

    mock_session = AsyncMock()

    # Create a proper async context manager mock
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_session
    mock_context_manager.__aexit__.return_value = None

    with patch.object(ingest.client, "session", return_value=mock_context_manager):
        count = await ingest.ingest_triples(triples)
        assert count == 0
