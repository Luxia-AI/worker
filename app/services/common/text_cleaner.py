"""
Text cleaning and normalization utilities for fact extraction and entity processing.
"""

import re
from typing import List


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and standardizing case.

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text (stripped, single spaces, lowercase)
    """
    if not isinstance(text, str):
        return ""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_statement(statement: str) -> str:
    """
    Clean a factual statement for storage and retrieval.

    Removes:
        - Leading/trailing whitespace
        - Extra internal whitespace
        - Common artifacts

    Args:
        statement: Fact statement text

    Returns:
        Cleaned statement
    """
    if not statement:
        return ""
    text = normalize_text(statement)
    # Remove markdown artifacts
    text = re.sub(r"[*_`#~]", "", text)
    # Remove extra punctuation
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


def truncate_content(content: str, max_length: int = 2000) -> str:
    """
    Truncate content while preserving word boundaries.

    Args:
        content: Full content text
        max_length: Maximum length in characters

    Returns:
        Truncated content
    """
    if len(content) <= max_length:
        return content
    truncated = content[:max_length]
    # Find last complete word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:  # At least 80% through
        return truncated[:last_space].strip() + "..."
    return truncated.strip() + "..."


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Text potentially containing HTML

    Returns:
        Text with HTML tags removed
    """
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    return text


def sanitize_entity(entity: str) -> str:
    """
    Clean and normalize entity strings for consistent matching.

    Args:
        entity: Raw entity string

    Returns:
        Sanitized entity (lowercase, trimmed, no extra spaces)
    """
    if not isinstance(entity, str):
        return ""
    entity = normalize_text(entity).lower()
    # Remove special characters but keep hyphens and parentheses
    entity = re.sub(r"[^\w\s\-\(\)]", "", entity)
    return entity.strip()


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text, handling common edge cases.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    if not text:
        return []
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Clean and filter
    cleaned = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return cleaned
