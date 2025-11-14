from sentence_transformers import SentenceTransformer

# Lazy initialization - only initialize when needed
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim model
    return _model