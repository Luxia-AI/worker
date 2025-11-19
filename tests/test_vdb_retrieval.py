from unittest.mock import MagicMock, patch

import pytest

from app.services.vdb.vdb_retrieval import VDBRetrieval


@pytest.mark.asyncio
@patch("app.services.vdb.vdb_retrieval.embed_async")
@patch("app.services.vdb.vdb_retrieval.get_pinecone_index")
async def test_vdb_retrieval(mock_get_index, mock_embed):
    mock_index = MagicMock()
    mock_get_index.return_value = mock_index

    # Fake embedding
    mock_embed.return_value = [[0.1, 0.2, 0.3]]

    # Fake Pinecone match results
    mock_index.query.return_value = {
        "matches": [
            {
                "score": 0.95,
                "metadata": {
                    "statement": "Vitamin D improves bone strength.",
                    "entities": ["vitamin d", "bone strength"],
                    "source_url": "https://example.com",
                },
            }
        ]
    }

    retriever = VDBRetrieval()
    results = await retriever.search("vitamin d bone", top_k=1)

    assert len(results) == 1  # nosec
    assert results[0]["score"] == 0.95  # nosec
    assert results[0]["statement"].startswith("Vitamin D")  # nosec
