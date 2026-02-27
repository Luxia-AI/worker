import asyncio
import os
import sys
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from app.constants.config import EMBEDDING_MODEL_NAME_PROD, EMBEDDING_MODEL_NAME_TEST
from app.core.logger import get_logger

logger = get_logger(__name__)

_model = None
_embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32") or 32)
_embedding_device = os.getenv("EMBEDDING_DEVICE", "").strip() or None

# Fallback model if primary fails (smaller, more reliable)
EMBEDDING_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"

# Set deterministic mode for reproducible embeddings
torch.set_num_threads(1)  # Single thread for consistent results
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
if hasattr(torch, "use_deterministic_algorithms"):
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass  # Some operations don't support deterministic mode


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
    Falls back to smaller model if primary fails.
    """
    global _model
    if _model is None:
        # Use test model if pytest is loaded, production model otherwise
        model_name = EMBEDDING_MODEL_NAME_TEST if _is_test_environment() else EMBEDDING_MODEL_NAME_PROD

        # Check if HuggingFace token is available
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        try:
            logger.info(f"Loading embedding model: {model_name}")
            _model = SentenceTransformer(model_name, token=hf_token, device=_embedding_device)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info(f"Falling back to {EMBEDDING_MODEL_FALLBACK}")
            try:
                _model = SentenceTransformer(EMBEDDING_MODEL_FALLBACK, token=hf_token, device=_embedding_device)
                logger.info(f"Fallback embedding model loaded: {EMBEDDING_MODEL_FALLBACK}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise RuntimeError(f"Could not load any embedding model: {e}, {e2}")
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
        embeddings = model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=max(1, _embedding_batch_size),
        )
        return [embedding.tolist() for embedding in embeddings]

    return await loop.run_in_executor(None, _encode)
