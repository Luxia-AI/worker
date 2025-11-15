from typing import Any, List, cast

import numpy as np

EMBEDDING_DIM: int = 1536  # Match Pinecone index dimension


def _pad_embedding(embedding: Any) -> List[Any]:
    """Pad embedding to match Pinecone index dimension."""
    emb_array = np.array(embedding)
    if len(emb_array) < EMBEDDING_DIM:
        padding = np.zeros(EMBEDDING_DIM - len(emb_array))
        emb_array = np.concatenate([emb_array, padding])
    return cast(List[Any], emb_array[:EMBEDDING_DIM].tolist())
