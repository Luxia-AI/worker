from typing import Optional

from sentence_transformers import SentenceTransformer

# Lazy initialization - only initialize when needed
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim model
    return _model
