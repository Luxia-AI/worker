from typing import Any, Dict, List

from app.constants.config import VDB_MIN_SCORE
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

    @staticmethod
    def _as_dict(response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        to_dict = getattr(response, "to_dict", None)
        if callable(to_dict):
            try:
                return to_dict()
            except Exception:
                return {}
        return {}

    async def search(self, query: str, top_k: int = 5, topics: List[str] | None = None) -> List[Dict[str, Any]]:
        """
        Search vector DB for semantically similar facts.

        Args:
            query: Search query text
            top_k: Number of top results to return
            topics: Required topic filter (no unrestricted queries)

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

        if not topics:
            logger.warning("[VDBRetrieval] No topics provided; skipping VDB retrieval")
            return []

        pinecone_filter: Dict[str, Any] = {"topic": {"$in": list(set(topics + ["other"]))}}
        if self.language:
            pinecone_filter["language"] = self.language

        try:
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
        if hasattr(response, "matches"):
            matches = response.matches or []
        else:
            matches = self._as_dict(response).get("matches") or []

        # Fallback: if strict topic filter produced no matches, retry with language-only filter.
        if not matches:
            try:
                fallback_filter: Dict[str, Any] = {}
                if self.language:
                    fallback_filter["language"] = self.language
                response = self.index.query(
                    vector=vector,
                    top_k=top_k,
                    namespace=self.namespace,
                    include_metadata=True,
                    filter=fallback_filter if fallback_filter else None,
                )
                matches = (
                    response.matches or []
                    if hasattr(response, "matches")
                    else self._as_dict(response).get("matches") or []
                )
            except Exception:
                pass

        results = []
        dropped = 0
        low_signal_phrases = (
            "data element definitions",
            "registration or results information",
            "javascript and cookies",
            "requires human verification",
        )
        for m in matches:
            if hasattr(m, "metadata"):
                metadata = m.metadata or {}
                score = float(getattr(m, "score", 0.0) or 0.0)
            else:
                metadata = m.get("metadata", {}) if isinstance(m, dict) else {}
                score = float((m.get("score", 0.0) if isinstance(m, dict) else 0.0) or 0.0)
            if score < VDB_MIN_SCORE:
                dropped += 1
                continue
            result = {
                "score": score,
                "statement": metadata.get("statement", ""),
                "entities": metadata.get("entities", []) or [],
                "source_url": metadata.get("source_url") or metadata.get("source", ""),
                "published_at": metadata.get("published_at"),
                "credibility": metadata.get("credibility"),
                "source": metadata.get("source"),
                "domain": metadata.get("domain"),
                "topic": metadata.get("topic"),
                "doc_type": metadata.get("doc_type"),
                "fact_type": metadata.get("fact_type"),
                "count_value": metadata.get("count_value"),
            }
            if result["statement"]:  # Only include non-empty statements
                if any(p in result["statement"].lower() for p in low_signal_phrases):
                    continue
                results.append(result)

        # DETERMINISTIC ORDERING: Sort by score DESC, then statement ASC for consistent results
        # This ensures identical queries return identical ordering even when scores are equal
        results.sort(key=lambda r: (-r["score"], r["statement"]))

        top_score = f"{results[0]['score']:.3f}" if results else "N/A"
        logger.debug(
            f"[VDBRetrieval] Query='{query[:50]}...' returned {len(results)} matches " f"(top score: {top_score})"
        )
        if dropped:
            logger.info(f"[VDBRetrieval] Dropped {dropped} matches below min score {VDB_MIN_SCORE}")
        return results

    def fetch_by_ids(self, ids: List[str], include_values: bool = False) -> List[Dict[str, Any]]:
        if not ids:
            return []
        try:
            try:
                response = self.index.fetch(ids=ids, namespace=self.namespace, include_values=include_values)
            except TypeError:
                response = self.index.fetch(ids=ids, namespace=self.namespace)
        except Exception as e:
            logger.error(f"[VDBRetrieval] Pinecone fetch failed: {e}")
            return []

        if hasattr(response, "vectors"):
            vectors = response.vectors or {}
        else:
            vectors = self._as_dict(response).get("vectors") or {}
        results = []
        for vec_id, vec_data in vectors.items():
            if hasattr(vec_data, "metadata"):
                metadata = vec_data.metadata or {}
                values = getattr(vec_data, "values", None)
            elif isinstance(vec_data, dict):
                metadata = vec_data.get("metadata", {})
                values = vec_data.get("values")
            else:
                metadata = {}
                values = None
            result = {
                "id": vec_id,
                "statement": metadata.get("statement", ""),
                "entities": metadata.get("entities", []) or [],
                "source_url": metadata.get("source_url") or metadata.get("source", ""),
                "published_at": metadata.get("published_at"),
                "credibility": metadata.get("credibility"),
                "source": metadata.get("source"),
                "domain": metadata.get("domain"),
                "topic": metadata.get("topic"),
                "doc_type": metadata.get("doc_type"),
                "fact_type": metadata.get("fact_type"),
                "count_value": metadata.get("count_value"),
            }
            if include_values:
                result["values"] = values
            if result["statement"]:
                results.append(result)

        return results
