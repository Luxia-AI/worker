"""
KG Triple Normalization Utility

Converts Knowledge Graph triples to textual statements that can be used as evidence
in the trust ranking system, maintaining consistency with VDB evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from app.services.ranking.trust_ranker import EvidenceItem


@dataclass
class KGTriple:
    """Represents a Knowledge Graph triple."""

    subject: str
    relation: str
    object: str
    source_url: Optional[str] = None
    published_at: Optional[str] = None
    confidence: Optional[float] = None


def triple_to_statement(triple: KGTriple) -> str:
    """
    Convert a KG triple to a stable textual statement.

    Format: "{subject} {relation} {object}."
    - Normalizes whitespace
    - Replaces underscores in relation with spaces
    """
    # Replace underscores with spaces in relation
    relation_normalized = triple.relation.replace("_", " ")

    # Create statement
    statement = f"{triple.subject} {relation_normalized} {triple.object}."

    # Normalize whitespace, preserving the final period
    parts = statement.rstrip(".").split()
    statement = " ".join(parts) + "."

    return statement


def triples_to_evidence(triples: List[KGTriple], semantic_score_provider: Callable[[str], float]) -> List[EvidenceItem]:
    """
    Convert KG triples to EvidenceItem list for trust ranking.

    Args:
        triples: List of KGTriple objects
        semantic_score_provider: Function that takes a statement string and returns float 0..1

    Returns:
        List of EvidenceItem with neutral stance initially
    """
    evidence_items = []

    for triple in triples:
        statement = triple_to_statement(triple)
        semantic_score = semantic_score_provider(statement)

        evidence_item = EvidenceItem(
            statement=statement,
            semantic_score=semantic_score,
            source_url=triple.source_url,
            published_at=triple.published_at,
            stance="neutral",  # Initially neutral, will be classified later
        )

        evidence_items.append(evidence_item)

    return evidence_items
