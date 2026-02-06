from typing import Any, Optional

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.logger import get_logger
from app.services.embedding.model import get_embedding_model

logger = get_logger(__name__)

_pc: Optional[Pinecone] = None
_index = None


def get_pinecone_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc


def get_pinecone_index() -> Any:
    global _index
    if _index:
        return _index

    # Get actual embedding dimension from ST model
    model = get_embedding_model()
    dimension = model.get_sentence_embedding_dimension()
    logger.info(f"[Pinecone] Embedding model dimension: {dimension}")

    pc = get_pinecone_client()
    index_name = settings.PINECONE_INDEX_NAME
    assert index_name is not None, "PINECONE_INDEX_NAME must be set"  # nosec B101

    existing_indexes = pc.list_indexes()
    existing_names = [idx.name for idx in existing_indexes]

    if index_name in existing_names:
        # Check if existing index has correct dimension
        index_info = next((idx for idx in existing_indexes if idx.name == index_name), None)
        if index_info and index_info.dimension != dimension:
            logger.warning(
                f"[Pinecone] Index '{index_name}' has dimension {index_info.dimension}, "
                f"but model has {dimension}. Deleting and recreating index..."
            )
            pc.delete_index(index_name)
            existing_names.remove(index_name)

    if index_name not in existing_names:
        logger.info(
            f"[Pinecone] Creating index '{index_name}' with dimension {dimension} "
            f"(cloud={settings.PINECONE_CLOUD}, region={settings.PINECONE_REGION})"
        )
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
        )

    _index = pc.Index(index_name)
    return _index
