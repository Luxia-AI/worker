import asyncio
import os
import sys
from typing import List

from sentence_transformers import SentenceTransformer

from app.constants.config import EMBEDDING_MODEL_NAME_PROD, EMBEDDING_MODEL_NAME_TEST
from app.core.logger import get_logger

logger = get_logger(__name__)

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
    Can be overridden with EMBEDDING_MODEL environment variable.
    """
    global _model
    if _model is None:
        # Allow environment override, otherwise use test/prod logic
        model_name = os.environ.get("EMBEDDING_MODEL")
        if not model_name:
            model_name = EMBEDDING_MODEL_NAME_TEST if _is_test_environment() else EMBEDDING_MODEL_NAME_PROD
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    return _model
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
