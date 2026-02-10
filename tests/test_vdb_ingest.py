from unittest.mock import MagicMock, patch

import pytest

from app.constants.config import ValidationState
from app.services.vdb.vdb_ingest import VDBIngest


@pytest.mark.asyncio
@patch("app.services.evidence_validator.EvidenceValidator.get_validation_state")
@patch("app.services.vdb.vdb_ingest.embed_async")
@patch("app.services.vdb.vdb_ingest.get_pinecone_index")
@patch("app.services.vdb.vdb_ingest.LexicalIndex.upsert_facts")
@patch("app.services.vdb.vdb_ingest.MetadataEnricher")
async def test_vdb_ingest(
    mock_enricher_cls,
    mock_lexical_upsert,
    mock_get_index,
    mock_embed,
    mock_validation,
):
    mock_index = MagicMock()
    mock_get_index.return_value = mock_index

    mock_embed.return_value = [[0.1, 0.2, 0.3]]  # fake embedding
    mock_validation.return_value = ValidationState.TRUSTED

    class _DummyEnricher:
        async def enrich_facts(self, facts):
            return facts

    mock_enricher_cls.return_value = _DummyEnricher()
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
    mock_lexical_upsert.assert_called_once()


@patch("app.services.vdb.vdb_ingest.get_pinecone_index")
@patch("app.services.vdb.vdb_ingest.MetadataEnricher")
def test_get_processed_urls_scopes_by_topic_and_respects_cap(mock_enricher_cls, mock_get_index):
    mock_index = MagicMock()
    mock_get_index.return_value = mock_index

    class _DummyEnricher:
        async def enrich_facts(self, facts):
            return facts

    mock_enricher_cls.return_value = _DummyEnricher()
    mock_index.list.return_value = [["id1", "id2", "id3"]]
    mock_index.fetch.return_value = {
        "vectors": {
            "id1": {"metadata": {"source_url": "https://example.org/a", "topic": "nutrition"}},
            "id2": {"metadata": {"source_url": "https://example.org/b", "topic": "cardio"}},
            "id3": {"metadata": {"source_url": "https://example.org/c", "topic": "other"}},
        }
    }

    ingest = VDBIngest()
    urls = ingest.get_processed_urls(topics=["nutrition"], max_urls=1)

    assert len(urls) == 1
    assert "https://example.org/a" in urls or "https://example.org/c" in urls
