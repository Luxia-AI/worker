from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.embedding.model import embed_async
from app.services.vdb.pinecone_client import get_pinecone_index

logger = get_logger(__name__)


class VDBRetrieval:
    """
    Semantic retrieval from Pinecone vector database.
    Returns facts with full metadata for hybrid ranking.
    """

    def __init__(self, namespace: str = "health", language: str | None = "en") -> None:
        self.index = get_pinecone_index()
        self.namespace = namespace
        self.language = language

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search vector DB for semantically similar facts.

        Args:
            query: Search query text
            top_k: Number of top results to return

        Returns:
            List of fact dicts with keys:
                - score: float (cosine similarity [0, 1])
                - statement: str (fact claim)
                - entities: List[str] (extracted entities)
                - source_url: str
                - published_at: Optional[str] (ISO format)
                - credibility: float [0, 1] (if available in metadata)
                - source: Optional[str] (alternative to source_url)
        """
        try:
            embedding = await embed_async([query])
            vector = embedding[0]
        except Exception as e:
            logger.error(f"[VDBRetrieval] Embedding generation failed: {e}")
            return []

        try:
            pinecone_filter = {"language": self.language} if self.language else None
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
                filter=pinecone_filter,
            )
        except Exception as e:
            logger.error(f"[VDBRetrieval] Pinecone query failed: {e}")
            return []

        matches = response.get("matches") or []

        results = []
        for m in matches:
            metadata = m.get("metadata", {})
            result = {
                "score": float(m.get("score", 0.0)),
                "statement": metadata.get("statement", ""),
                "entities": metadata.get("entities", []) or [],
                "source_url": metadata.get("source_url") or metadata.get("source", ""),
                "published_at": metadata.get("published_at"),
                "credibility": metadata.get("credibility"),
                "source": metadata.get("source"),
            }
            if result["statement"]:  # Only include non-empty statements
                results.append(result)

        # DETERMINISTIC ORDERING: Sort by score DESC, then statement ASC for consistent results
        # This ensures identical queries return identical ordering even when scores are equal
        results.sort(key=lambda r: (-r["score"], r["statement"]))

        top_score = f"{results[0]['score']:.3f}" if results else "N/A"
        logger.debug(
            f"[VDBRetrieval] Query='{query[:50]}...' returned {len(results)} matches " f"(top score: {top_score})"
        )
        return results
