from unittest.mock import MagicMock, patch

import pytest

from app.constants.config import ValidationState
from app.services.vdb.vdb_ingest import VDBIngest


@pytest.mark.asyncio
@patch("app.services.evidence_validator.EvidenceValidator.get_validation_state")
@patch("app.services.vdb.vdb_ingest.embed_async")
@patch("app.services.vdb.vdb_ingest.get_pinecone_index")
async def test_vdb_ingest(mock_get_index, mock_embed, mock_validation):
    mock_index = MagicMock()
    mock_get_index.return_value = mock_index

    mock_embed.return_value = [[0.1, 0.2, 0.3]]  # fake embedding
    mock_validation.return_value = ValidationState.TRUSTED

    ingest = VDBIngest()

    facts = [
        {
            "fact_id": "f1",
            "statement": "Vitamin D improves immunity.",
            "entities": ["vitamin d", "immunity"],
            "source_url": "https://example.com",
        }
    ]

    ids = await ingest.embed_and_ingest(facts)

    assert ids == ["f1"]  # nosec
    mock_embed.assert_called_once()
    mock_index.upsert.assert_called_once()
