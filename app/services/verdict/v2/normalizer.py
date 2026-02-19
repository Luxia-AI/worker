from __future__ import annotations

import re

_BLOCKED_PATTERNS = (
    r"\baccess denied\b",
    r"\bdenied access\b",
    r"\bpermission required\b",
    r"\bforbidden\b",
    r"\b403\b",
    r"\bcaptcha\b",
    r"\bblocked\b",
    r"\btemporarily unavailable\b",
)


def is_blocked_content(statement: str) -> bool:
    text = str(statement or "").strip().lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in _BLOCKED_PATTERNS)


def normalize_relevance_label(relevance: str) -> str:
    label = str(relevance or "").strip().upper()
    if label in {"SUPPORTS", "VALID", "PARTIALLY_VALID", "PARTIALLY_SUPPORTS"}:
        return "SUPPORTS"
    if label in {"REFUTES", "CONTRADICTS", "INVALID", "PARTIALLY_INVALID", "PARTIALLY_CONTRADICTS"}:
        return "REFUTES"
    return "NEUTRAL"
