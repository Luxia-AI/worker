"""
Advanced deduplication strategies for facts, statements, and candidates.
"""

from typing import Any, Dict, List, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)


def dedup_by_semantic_similarity(
    items: List[Dict[str, Any]], text_key: str, similarity_threshold: float = 0.95
) -> List[Dict[str, Any]]:
    """
    Deduplicate items by semantic text similarity (basic approach: string matching + length).

    Args:
        items: List of dicts with text content
        text_key: Key containing text to compare
        similarity_threshold: Threshold for considering texts similar (0-1)

    Returns:
        Deduplicated list keeping first occurrence
    """

    def normalized_text(text: str) -> str:
        """Normalize text for comparison."""
        return " ".join(text.lower().split())

    seen_texts: List[str] = []
    result = []

    for item in items:
        text = item.get(text_key, "")
        norm_text = normalized_text(text)

        is_duplicate = False
        for seen in seen_texts:
            if norm_text == seen:
                is_duplicate = True
                break

        if not is_duplicate:
            seen_texts.append(norm_text)
            result.append(item)

    return result


def dedup_facts_by_statement_and_source(
    facts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deduplicate facts by statement + source_url combination.

    Keeps fact with highest confidence if duplicates exist.

    Args:
        facts: List of fact dicts

    Returns:
        Deduplicated facts
    """
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for fact in facts:
        statement = fact.get("statement", "").lower().strip()
        source = fact.get("source_url", "")
        confidence = fact.get("confidence", 0.5)

        key = (statement, source)

        if key not in seen or seen[key].get("confidence", 0) < confidence:
            seen[key] = fact

    return list(seen.values())


def dedup_triples_by_structure(
    triples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deduplicate triples by (subject, relation, object, source_url).

    Keeps triple with highest confidence if duplicates exist.

    Args:
        triples: List of triple dicts with subject, relation, object

    Returns:
        Deduplicated triples
    """
    seen: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    for triple in triples:
        subject = triple.get("subject", "").lower().strip()
        relation = triple.get("relation", "").lower().strip()
        obj = triple.get("object", "").lower().strip()
        source = triple.get("source_url", "")
        confidence = triple.get("confidence", 0.5)

        key = (subject, relation, obj, source)

        if key not in seen or seen[key].get("confidence", 0) < confidence:
            seen[key] = triple

    return list(seen.values())


def dedup_urls(urls: List[str]) -> List[str]:
    """
    Deduplicate URLs after normalization.
    Note: This is now provided via url_helpers.dedup_urls for clarity.

    Args:
        urls: List of URLs

    Returns:
        Deduplicated URLs
    """
    # Import here to avoid circular imports
    from app.services.common.url_helpers import dedup_urls as dedup_urls_impl

    return dedup_urls_impl(urls)


def dedup_candidates_by_score(
    candidates: List[Dict[str, Any]],
    statement_key: str = "statement",
    source_key: str = "source_url",
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """
    Deduplicate candidates by statement + source, keeping highest score.

    Args:
        candidates: List of candidate dicts
        statement_key: Key for statement/text
        source_key: Key for source/URL
        score_key: Key for numeric score

    Returns:
        Deduplicated candidates
    """
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for candidate in candidates:
        statement = candidate.get(statement_key, "").lower().strip()
        source = candidate.get(source_key, "")
        score = candidate.get(score_key, 0.0)

        key = (statement, source)

        if key not in seen or seen[key].get(score_key, 0) < score:
            seen[key] = candidate

    return list(seen.values())


def merge_duplicate_candidates(
    candidates: List[Dict[str, Any]], statement_key: str = "statement"
) -> List[Dict[str, Any]]:
    """
    Merge candidate lists when same statement appears multiple times.

    Aggregates sources and averages scores.

    Args:
        candidates: List of candidates
        statement_key: Key for statement text

    Returns:
        Merged candidates
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for candidate in candidates:
        statement = candidate.get(statement_key, "").lower().strip()

        if statement not in grouped:
            grouped[statement] = candidate.copy()
            # Initialize sources list
            if "sources" not in grouped[statement]:
                grouped[statement]["sources"] = []
            if "source_url" in candidate:
                grouped[statement]["sources"].append(candidate["source_url"])
        else:
            # Merge sources
            if "source_url" in candidate:
                sources = grouped[statement].get("sources", [])
                if candidate["source_url"] not in sources:
                    sources.append(candidate["source_url"])
                grouped[statement]["sources"] = sources

            # Average scores if present
            if "score" in candidate:
                old_score = grouped[statement].get("score", 0.0)
                new_score = (old_score + candidate["score"]) / 2
                grouped[statement]["score"] = new_score

    return list(grouped.values())
