from typing import Any, Dict, List

from app.services.embedding import _pad_embedding
from app.services.llms.embedding_llm import _get_model
from app.services.vector_db import _get_index


def retrieve_similar(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    model = _get_model()
    query_embedding = model.encode([query])
    padded_embedding = _pad_embedding(query_embedding[0])
    index = _get_index()
    results = index.query(vector=padded_embedding, top_k=top_k, include_metadata=True)
    return [{"score": result["score"], "text": result["metadata"]["text"]} for result in results["matches"]]
