import asyncio
from typing import List

from sentence_transformers import SentenceTransformer

# Recommended for RAG search -- strong performance
EMBEDDING_MODEL_NAME = "sentence-transformers/multilingual-e5-large"

_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy init the embedding model.
    Loads once and reuses for all workers.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


async def embed_async(sentences: List[str]) -> List[List[float]]:
    """
    Async wrapper for embedding generation.
    Runs CPU/Threadpool operations off main event loop.
    """
    loop = asyncio.get_event_loop()
    model = get_embedding_model()

    return await loop.run_in_executor(None, lambda: model.encode(sentences, convert_to_numpy=False))
