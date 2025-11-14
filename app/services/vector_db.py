from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings

from app.services.llm import _get_model
from app.services.embedding import _pad_embedding, EMBEDDING_DIM

INDEX_NAME = settings.pinecone_index_name or "rag-research-index-test"  # Default index name

# Lazy initialization - only initialize when needed
_pc = None
_index = None

def _get_pinecone_client():
    global _pc
    if _pc is None:
        api_key = settings.pinecone_api_key
        _pc = Pinecone(api_key=api_key)
    return _pc

def _get_index():
    global _index
    if _index is None:
        pc = _get_pinecone_client()
        if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(region="us-east-1", cloud="aws"),
            )
        _index = pc.Index(INDEX_NAME)
    return _index