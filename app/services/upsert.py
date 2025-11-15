from typing import Any, Dict, List

from app.services.embedding import _pad_embedding
from app.services.llm import _get_model
from app.services.vector_db import _get_index


def insert_dummy_data() -> int:
    texts: List[str] = [
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "The chemical symbol for water is H2O.",
        "The Earth revolves around the Sun.",
        "COVID-19 vaccines are effective for preventing severe illness.",
        "Elon Musk founded SpaceX.",
        "Artificial intelligence can detect misinformation.",
        "Chocolate improves cognitive performance.",
    ]
    model = _get_model()
    embeddings = model.encode(texts)
    vectors: List[Dict[str, Any]] = [
        {"id": f"vec{i+1}", "values": _pad_embedding(emb), "metadata": {"text": text}}
        for i, (emb, text) in enumerate(zip(embeddings, texts))
    ]
    index = _get_index()
    index.upsert(vectors)
    return len(vectors)
