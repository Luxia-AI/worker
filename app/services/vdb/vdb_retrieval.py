import re
from typing import Any, Dict, List

from app.constants.config import PIPELINE_MIN_RANK_CANDIDATES, VDB_BACKFILL_MIN_SCORE, VDB_MIN_SCORE
from app.core.logger import get_logger
from app.services.embedding.model import embed_async
from app.services.vdb.pinecone_client import get_pinecone_index

logger = get_logger(__name__)


def normalize_query_for_embedding(text: str) -> str:
    """
    Normalize query text for semantic embedding retrieval.

    Removes web-search operators and noisy filter tokens so only semantic text
    is embedded and sent to vector retrieval.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    cleaned = raw
    cleaned = re.sub(
        r"(?<!\S)-(?:facebook|quora|reddit|pinterest|youtube|testimonial|opinion)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bsite:[^\s]+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace('"', " ")
    cleaned = re.sub(r"[`~!@#$%^&*_=+{}\[\]|\\<>]", " ", cleaned)
    cleaned = re.sub(r"[(),;:]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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

    async def search(
        self,
        query: str,
        top_k: int = 5,
        topics: List[str] | None = None,
        min_score: float | None = None,
    ) -> List[Dict[str, Any]]:
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
        normalized_query = normalize_query_for_embedding(query)
        if not normalized_query:
            return []

        try:
            embedding = await embed_async([normalized_query])
            vector = embedding[0]
        except Exception as e:
            logger.error(f"[VDBRetrieval] Embedding generation failed: {e}")
            return []

        topic_list = [t for t in (topics or []) if t]
        pinecone_filter: Dict[str, Any] = {}
        if topic_list:
            pinecone_filter["topic"] = {"$in": list(set(topic_list + ["other"]))}
        if self.language:
            pinecone_filter["language"] = self.language

        try:
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else None,
            )
        except Exception as e:
            logger.error(f"[VDBRetrieval] Pinecone query failed: {e}")
            return []
        if hasattr(response, "matches"):
            matches = response.matches or []
        else:
            matches = self._as_dict(response).get("matches") or []

        # Fallback: if strict topic filter produced no matches, retry with language-only filter.
        if topic_list and not matches:
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
        below_threshold: List[Dict[str, Any]] = []
        dropped = 0
        dropped_low_signal = 0
        low_signal_phrases = (
            "data element definitions",
            "registration or results information",
            "javascript and cookies",
            "requires human verification",
        )
        effective_min_score = float(min_score if min_score is not None else VDB_MIN_SCORE)
        pre_filter_count = len(matches)
        for m in matches:
            if hasattr(m, "metadata"):
                metadata = m.metadata or {}
                score = float(getattr(m, "score", 0.0) or 0.0)
            else:
                metadata = m.get("metadata", {}) if isinstance(m, dict) else {}
                score = float((m.get("score", 0.0) if isinstance(m, dict) else 0.0) or 0.0)
            if score < effective_min_score:
                dropped += 1
                # Keep a lightweight backfill pool to avoid starvation.
                if score >= VDB_BACKFILL_MIN_SCORE:
                    below_threshold.append(
                        {
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
                    )
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
                    dropped_low_signal += 1
                    continue
                results.append(result)

        # Backfill if too few candidates survive strict threshold.
        if len(results) < PIPELINE_MIN_RANK_CANDIDATES and below_threshold:
            needed = PIPELINE_MIN_RANK_CANDIDATES - len(results)
            below_threshold.sort(key=lambda r: (-r["score"], r["statement"]))
            for item in below_threshold[: max(0, needed)]:
                if item["statement"] and not any(
                    p in item["statement"].lower() for p in ("javascript and cookies", "human verification")
                ):
                    results.append(item)

        # DETERMINISTIC ORDERING: Sort by score DESC, then statement ASC for consistent results
        # This ensures identical queries return identical ordering even when scores are equal
        results.sort(key=lambda r: (-r["score"], r["statement"]))

        top_score = f"{results[0]['score']:.3f}" if results else "N/A"
        logger.debug(
            "[VDBRetrieval] Query='%s' normalized='%s' returned %d matches (top score: %s)",
            query[:50],
            normalized_query[:50],
            len(results),
            top_score,
        )
        logger.info(
            "[VDBRetrieval][Filter] query='%s' pre=%d above_min=%d post=%d "
            "dropped_min=%d dropped_low_signal=%d min=%.2f",
            normalized_query[:80],
            pre_filter_count,
            max(0, pre_filter_count - dropped),
            len(results),
            dropped,
            dropped_low_signal,
            effective_min_score,
        )
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
