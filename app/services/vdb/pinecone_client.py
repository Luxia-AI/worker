from typing import Optional

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.services.embedding.model import get_embedding_model

_pc: Optional[Pinecone] = None
_index = None


def get_pinecone_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_pinecone_index():
    global _index
    if _index:
        return _index

    # Get actual embedding dimension from ST model
    model = get_embedding_model()
    dimension = model.get_sentence_embedding_dimension()

    pc = get_pinecone_client()
    index_name = settings.pinecone_index_name

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    _index = pc.Index(index_name)
    return _index
