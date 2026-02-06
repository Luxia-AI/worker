import re
from typing import Any, Dict, List, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)


_COUNT_TOKENS = {
    "bones",
    "bone",
    "beats",
    "beat",
    "cells",
    "cell",
    "cases",
    "case",
    "times",
    "bpm",
    "count",
}


def _has_number(text: str) -> bool:
    return bool(re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text or ""))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", (text or "").lower())


def is_count_claim(claim: str) -> bool:
    text = (claim or "").lower()
    if re.search(r"\bhow many\b|\bnumber of\b", text):
        return True
    return _has_number(text) and any(tok in text for tok in _COUNT_TOKENS)


def _entity_overlap(statement: str, claim_entities: List[str]) -> int:
    if not statement or not claim_entities:
        return 0
    stmt_tokens = set(_tokenize(statement))
    ent_tokens = set()
    for ent in claim_entities:
        ent_tokens.update(_tokenize(ent))
    return len(stmt_tokens & ent_tokens)


def filter_candidates_for_count_claim(
    semantic_candidates: List[Dict[str, Any]],
    kg_candidates: List[Dict[str, Any]],
    claim_text: str,
    claim_entities: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not is_count_claim(claim_text):
        return semantic_candidates, kg_candidates

    filtered_sem = []
    for cand in semantic_candidates:
        stmt = cand.get("statement") or ""
        fact_type = cand.get("fact_type", "")
        if not _has_number(stmt):
            continue
        if fact_type == "count" or _entity_overlap(stmt, claim_entities) >= 1:
            filtered_sem.append(cand)

    filtered_kg = []
    for cand in kg_candidates:
        stmt = cand.get("statement") or ""
        if not _has_number(stmt):
            continue
        if _entity_overlap(stmt, claim_entities) >= 1:
            filtered_kg.append(cand)

    logger.info(
        "[EvidenceGate] Count-claim filter: %d->%d semantic, %d->%d kg",
        len(semantic_candidates),
        len(filtered_sem),
        len(kg_candidates),
        len(filtered_kg),
    )
    return filtered_sem, filtered_kg
