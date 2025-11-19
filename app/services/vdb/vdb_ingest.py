from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.embedding.model import embed_async
from app.services.vdb.pinecone_client import get_pinecone_index

logger = get_logger(__name__)


class VDBIngest:
    def __init__(self, namespace: str = "health"):
        self.index = get_pinecone_index()
        self.namespace = namespace

    async def embed_and_ingest(self, facts: List[Dict[str, Any]]) -> List[str]:
        if not facts:
            return []

        logger.info(f"[VDBIngest] Embedding {len(facts)} facts")

        statements = [f["statement"] for f in facts]
        embeddings = await embed_async(statements)

        vectors = []
        for fact, emb in zip(facts, embeddings):
            vectors.append(
                {
                    "id": fact["fact_id"],
                    "values": emb,
                    "metadata": {
                        "statement": fact["statement"],
                        "entities": fact.get("entities", []),
                        "source_url": fact.get("source_url"),
                    },
                }
            )

        logger.info(f"[VDBIngest] Upserting into Pinecone: {len(vectors)} vectors")

        self.index.upsert(vectors=vectors, namespace=self.namespace)

        return [v["id"] for v in vectors]
