import pytest

from app.services.embedding.model import embed_async, get_embedding_model


@pytest.mark.asyncio
async def test_embedding_async_basic():
    sentences = ["Vitamin D improves bone health."]
    vectors = await embed_async(sentences)

    assert len(vectors) == 1  # nosec
    assert isinstance(vectors[0], list)  # nosec
    # Verify dimension matches the actual model
    expected_dim = get_embedding_model().get_sentence_embedding_dimension()
    assert len(vectors[0]) == expected_dim  # nosec
