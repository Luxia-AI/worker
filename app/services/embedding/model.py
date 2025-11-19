import asyncio
import sys
from typing import List

from sentence_transformers import SentenceTransformer

# Production model: recommended for RAG search -- strong performance
EMBEDDING_MODEL_NAME_PROD = "sentence-transformers/multilingual-e5-large"

# Test model: lightweight, fast downloads for testing
EMBEDDING_MODEL_NAME_TEST = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def _is_test_environment() -> bool:
    """
    Detect if we're running in a test environment.
    Checks for pytest in the module stack.
    """
    return "pytest" in sys.modules


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy init the embedding model.
    Loads once and reuses for all workers.
    Uses lightweight model for testing, production model otherwise.
    """
    global _model
    if _model is None:
        # Use test model if pytest is loaded, production model otherwise
        model_name = EMBEDDING_MODEL_NAME_TEST if _is_test_environment() else EMBEDDING_MODEL_NAME_PROD
        _model = SentenceTransformer(model_name)
    return _model


async def embed_async(sentences: List[str]) -> List[List[float]]:
    """
    Async wrapper for embedding generation.
    Runs CPU/Threadpool operations off main event loop.
    Returns embeddings as lists of floats.
    """
    loop = asyncio.get_event_loop()
    model = get_embedding_model()

    def _encode() -> List[List[float]]:
        embeddings = model.encode(sentences, convert_to_tensor=True)
        # Convert tensors to lists of floats
        return [embedding.tolist() for embedding in embeddings]

    return await loop.run_in_executor(None, _encode)
