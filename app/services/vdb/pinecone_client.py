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

    pc = get_pinecone_client()
    index_name = settings.PINECONE_INDEX_NAME
    assert index_name is not None, "PINECONE_INDEX_NAME must be set"  # nosec B101

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    _index = pc.Index(index_name)
    return _index
