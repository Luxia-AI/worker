from typing import Any, Dict, List

from app.services.embedding.model import embed_async
from app.services.vdb.pinecone_client import get_pinecone_index


class VDBRetrieval:
    def __init__(self, namespace="health"):
        self.index = get_pinecone_index()
        self.namespace = namespace

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = await embed_async([query])
        vector = embedding[0]

        response = self.index.query(vector=vector, top_k=top_k, namespace=self.namespace, include_metadata=True)

        matches = response.get("matches") or []

        return [
            {
                "score": m["score"],
                "statement": m["metadata"]["statement"],
                "entities": m["metadata"]["entities"],
                "source_url": m["metadata"]["source_url"],
            }
            for m in matches
        ]
