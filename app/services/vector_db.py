from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_NAME = settings.pinecone_index_name or "rag-research-index-test"  # Default index name
EMBEDDING_DIM = 1536  # Match Pinecone index dimension

# Lazy initialization - only initialize when needed
_pc = None
_index = None
_model = None

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

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim model
    return _model

def _pad_embedding(embedding):
    """Pad embedding to match Pinecone index dimension."""
    emb_array = np.array(embedding)
    if len(emb_array) < EMBEDDING_DIM:
        padding = np.zeros(EMBEDDING_DIM - len(emb_array))
        emb_array = np.concatenate([emb_array, padding])
    return emb_array[:EMBEDDING_DIM].tolist()

def insert_dummy_data():
    texts = [
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "The chemical symbol for water is H2O.",
        "The Earth revolves around the Sun.",
        "COVID-19 vaccines are effective for preventing severe illness.",
        "Elon Musk founded SpaceX.",
        "Artificial intelligence can detect misinformation.",
        "Chocolate improves cognitive performance."
    ]
    model = _get_model()
    embeddings = model.encode(texts)
    vectors = [
        {"id": f"vec{i+1}", "values": _pad_embedding(emb), "metadata": {"text": text}}
        for i, (emb, text) in enumerate(zip(embeddings, texts))
    ]
    index = _get_index()
    index.upsert(vectors)
    return len(vectors)

def retrieve_similar(query: str, top_k: int = 3):
    model = _get_model()
    query_embedding = model.encode([query])
    padded_embedding = _pad_embedding(query_embedding[0])
    index = _get_index()
    results = index.query(
        vector=padded_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [
        {
            "score": result["score"],
            "text": result["metadata"]["text"]
        }
        for result in results["matches"]
    ]