from unittest.mock import MagicMock, patch

import pytest

from app.services.embedding.model import embed_async


@pytest.mark.asyncio
@patch("app.services.embedding.model.get_embedding_model")
async def test_embedding_async_basic(mock_get_model):
    # Mock the embedding model to avoid downloading it
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]  # Mock embedding vector
    mock_model.get_sentence_embedding_dimension.return_value = 5
    mock_get_model.return_value = mock_model

    sentences = ["Vitamin D improves bone health."]
    vectors = await embed_async(sentences)

    assert len(vectors) == 1  # nosec
    assert isinstance(vectors[0], list)  # nosec
    assert len(vectors[0]) == 5  # nosec
