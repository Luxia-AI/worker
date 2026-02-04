"""
Trust Ranking Module: Advanced evidence ranking with component scores,
stance analysis, and post-level trust aggregation.

This module implements a comprehensive trust ranking system that evaluates evidence quality through multiple signals:
- Semantic similarity
- Source credibility
- Publication recency
- Stance/entailment analysis

It provides both per-evidence trust scores and post-level aggregation for decision-making.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse

from app.core.logger import get_logger

logger = get_logger(__name__)


# Default source credibility scores (configurable)
DEFAULT_SOURCE_SCORES = {
    "who.int": 0.95,
    "cdc.gov": 0.93,
    "nih.gov": 0.90,
    "pubmed.ncbi.nlm.nih.gov": 0.92,
    "nature.com": 0.90,
    "sciencedirect.com": 0.88,
    "bbc.com": 0.75,
    "reuters.com": 0.80,
    # Add more as needed
}


@dataclass
class EvidenceItem:
    """Represents a single piece of evidence for ranking."""

    statement: str
    semantic_score: float  # S_semantic: 0..1
    source_url: str
    published_at: Optional[str] = None  # ISO datetime string
    stance: str = "neutral"  # "entails", "contradicts", "neutral"
    trust: float = 0.0  # Computed trust score


class StanceClassifier(Protocol):
    """Protocol for stance/entailment classification."""

    def classify_stance(self, claim: str, evidence: str) -> str:
        """
        Classify the stance of evidence relative to claim.

        Returns:
            "entails", "contradicts", or "neutral"
        """
        ...


class DummyStanceClassifier:
    """Dummy implementation of stance classifier for testing/development."""

    def classify_stance(self, claim: str, evidence: str) -> str:
        """Simple heuristic-based stance classification."""
        evidence_lower = evidence.lower()
        claim_lower = claim.lower()

        # Simple heuristics
        if any(word in evidence_lower for word in ["not", "false", "incorrect", "debunked"]) and any(
            word in evidence_lower for word in claim_lower.split()
        ):
            return "contradicts"
        elif any(word in evidence_lower for word in ["supports", "confirms", "true", "accurate"]) and any(
            word in evidence_lower for word in claim_lower.split()
        ):
            return "entails"
        else:
            return "neutral"


class TrustRankingModule:
    """
    Advanced trust ranking module for evidence evaluation.

    Provides component-based scoring, ranking, deduplication, and post-level trust aggregation.
    """

    def __init__(
        self, stance_classifier: Optional[StanceClassifier] = None, source_scores: Optional[Dict[str, float]] = None
    ):
        self.stance_classifier = stance_classifier or DummyStanceClassifier()
        self.source_scores = source_scores or DEFAULT_SOURCE_SCORES.copy()

    def compute_source_score(self, source_url: str) -> float:
        """Compute source credibility score from domain."""
        try:
            domain = urlparse(source_url).netloc.lower()
            return self.source_scores.get(domain, 0.5)
        except Exception:
            return 0.5

    def compute_recency_score(self, published_at: Optional[str]) -> float:
        """Compute recency score based on publication date."""
        if not published_at:
            return 0.5

        try:
            # Handle various ISO formats
            dt_str = published_at.replace("Z", "+00:00")
            if dt_str.endswith("+00:00"):
                dt = datetime.fromisoformat(dt_str)
            else:
                dt = datetime.fromisoformat(dt_str + "+00:00")

            now = datetime.now(dt.tzinfo)
            days_diff = (now - dt).days

            if days_diff < 180:
                return 1.0
            elif days_diff < 365:
                return 0.8
            elif days_diff < 3 * 365:
                return 0.6
            elif days_diff < 5 * 365:
                return 0.4
            else:
                return 0.3
        except Exception:
            return 0.5

    def compute_stance_score(self, stance: str) -> float:
        """Convert stance string to numerical score."""
        stance_scores = {"entails": 1.0, "neutral": 0.5, "contradicts": -1.0}
        return stance_scores.get(stance, 0.5)

    def compute_trust_evidence(self, item: EvidenceItem) -> float:
        """
        Compute trust score for a single evidence item.

        Trust = w_semantic*S_semantic + w_source*S_source + w_recency*S_recency + w_stance*stance_mapped
        where stance_mapped = (S_stance + 1) / 2  # Map -1..1 to 0..1
        """
        S_semantic = item.semantic_score
        S_source = self.compute_source_score(item.source_url)
        S_recency = self.compute_recency_score(item.published_at)
        S_stance = self.compute_stance_score(item.stance)

        # Map stance to [0,1]
        stance_mapped = (S_stance + 1) / 2

        # Weighted sum with default weights
        w_semantic = 0.30
        w_source = 0.30
        w_recency = 0.20
        w_stance = 0.20

        trust = w_semantic * S_semantic + w_source * S_source + w_recency * S_recency + w_stance * stance_mapped

        # Clamp to [0,1]
        return max(0.0, min(trust, 1.0))

    def rank_evidence(self, evidence_list: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Rank evidence by trust score and deduplicate.

        Returns sorted list (trust desc, then semantic desc) with duplicates removed.
        """
        # Compute trust for each item
        for item in evidence_list:
            item.trust = self.compute_trust_evidence(item)

        # Sort by trust desc, then semantic desc
        evidence_list.sort(key=lambda x: (-x.trust, -x.semantic_score))

        # Deduplicate by normalized URL or text hash
        seen = set()
        deduped = []

        for item in evidence_list:
            # Use normalized URL as key, fallback to statement hash
            if item.source_url:
                key = item.source_url.lower().strip()
            else:
                key = hashlib.md5(item.statement.encode("utf-8")).hexdigest()

            if key not in seen:
                seen.add(key)
                deduped.append(item)

        logger.info(f"[TrustRankingModule] Ranked and deduped {len(evidence_list)} → {len(deduped)} evidence items")
        return deduped

    def compute_post_trust(self, ranked_evidence: List[Dict[str, Any]], top_k: int = 10) -> Dict[str, Any]:
        """
        Compute post-level trust metrics from top-k evidence.

        Args:
            ranked_evidence: List of evidence dicts with keys: fact, source, score, publish_date
            top_k: Number of top items to consider

        Returns:
            {
                "trust_post": float,  # Weighted mean trust
                "agreement_ratio": float,  # Fraction not contradicting
                "trust_grade": str  # A+, A, B, C, D, F
            }
        """
        # Convert dicts to EvidenceItem objects if needed
        evidence_items = []
        for item in ranked_evidence[:top_k]:
            if isinstance(item, dict):
                # Convert dict to EvidenceItem
                evidence_item = EvidenceItem(
                    statement=item.get("fact", ""),
                    semantic_score=item.get("score", 0.0),
                    source_url=item.get("source", ""),
                    published_at=item.get("publish_date"),
                    stance="neutral",  # Default stance, will be classified later if needed
                )
                # Compute trust for this item
                evidence_item.trust = self.compute_trust_evidence(evidence_item)
                evidence_items.append(evidence_item)
            else:
                evidence_items.append(item)

        if not evidence_items:
            return {"trust_post": 0.0, "agreement_ratio": 0.0, "trust_grade": "F"}

        trusts = [item.trust for item in evidence_items]

        # Self-weighted mean: sum(trust_i * trust_i) / sum(trust_i)
        if sum(trusts) > 0:
            trust_post = sum(t * t for t in trusts) / sum(trusts)
        else:
            trust_post = 0.0

        # Agreement ratio: fraction where stance != "contradicts"
        agreement_ratio = sum(1 for item in evidence_items if item.stance != "contradicts") / len(evidence_items)

        # Assign grade
        if trust_post >= 0.85:
            grade = "A+"
        elif trust_post >= 0.75:
            grade = "A"
        elif trust_post >= 0.65:
            grade = "B"
        elif trust_post >= 0.50:
            grade = "C"
        elif trust_post >= 0.35:
            grade = "D"
        else:
            grade = "F"

        return {"trust_post": trust_post, "agreement_ratio": agreement_ratio, "trust_grade": grade}

    def decide(self, evidence_list: List[EvidenceItem], threshold: float = 0.75) -> str:
        """
        Make decision based on post-level trust.

        Returns:
            "proceed_to_generation" if trust_post >= threshold
            "trigger_corrective_retrieval" otherwise
        """
        ranked = self.rank_evidence(evidence_list)
        post_trust = self.compute_post_trust(ranked)

        if post_trust["trust_post"] >= threshold:
            decision = "proceed_to_generation"
        else:
            decision = "trigger_corrective_retrieval"

        logger.info(
            f"[TrustRankingModule] Decision: {decision} "
            f"(trust_post={post_trust['trust_post']:.3f}, threshold={threshold})"
        )
        return decision

    def classify_stance_for_evidence(self, claim: str, evidence_list: List[EvidenceItem]) -> None:
        """
        Classify stance for evidence items that don't have it set.

        Modifies evidence_list in-place.
        """
        for item in evidence_list:
            if not item.stance or item.stance == "neutral":  # Only classify if not already set or neutral
                item.stance = self.stance_classifier.classify_stance(claim, item.statement)


# Backward compatibility: Keep the old TrustRanker class
class TrustRanker:
    """
    Legacy class for grade assignment. Use TrustRankingModule for new functionality.
    """

    GRADES = ["A+", "A", "B", "C", "D", "F"]

    @staticmethod
    def assign_grade(final_score: float) -> str:
        """Map final_score [0.0, 1.0] to a trust grade."""
        final_score = max(0.0, min(final_score, 1.0))

        if final_score >= 0.90:
            return "A+"
        elif final_score >= 0.80:
            return "A"
        elif final_score >= 0.70:
            return "B"
        elif final_score >= 0.60:
            return "C"
        elif final_score >= 0.50:
            return "D"
        else:
            return "F"

    @staticmethod
    def assign_semantic_confidence(semantic_score: float) -> str:
        """Assess semantic similarity confidence level."""
        semantic_score = max(0.0, min(semantic_score, 1.0))

        if semantic_score >= 0.8:
            return "HIGH"
        elif semantic_score >= 0.6:
            return "GOOD"
        elif semantic_score >= 0.4:
            return "FAIR"
        elif semantic_score >= 0.2:
            return "LOW"
        else:
            return "NONE"

    @staticmethod
    def enrich_ranked_results(ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add trust grades and semantic confidence to ranked results."""
        enriched = []

        for result in ranked_results:
            final_score = result.get("final_score", 0.0)
            sem_score = result.get("sem_score", 0.0)
            credibility = result.get("credibility", 0.5)
            entity_overlap = result.get("entity_overlap", 0.0)
            recency = result.get("recency", 0.0)

            grade = TrustRanker.assign_grade(final_score)
            sem_confidence = TrustRanker.assign_semantic_confidence(sem_score)

            # Build rationale
            factors = []
            if sem_score >= 0.6:
                factors.append(f"high semantic match ({sem_score:.2f})")
            elif sem_score >= 0.4:
                factors.append(f"fair semantic match ({sem_score:.2f})")

            if credibility >= 0.75:
                factors.append(f"trusted source ({credibility:.2f})")
            elif credibility >= 0.5:
                factors.append(f"moderate credibility ({credibility:.2f})")

            if entity_overlap >= 0.7:
                factors.append(f"strong entity overlap ({entity_overlap:.2f})")
            elif entity_overlap >= 0.3:
                factors.append(f"some entity match ({entity_overlap:.2f})")

            if recency >= 0.5:
                factors.append(f"recent source ({recency:.2f})")

            rationale = "; ".join(factors) if factors else "low confidence across all signals"

            enriched.append(
                {
                    **result,
                    "grade": grade,
                    "semantic_confidence": sem_confidence,
                    "grade_rationale": rationale,
                }
            )

        logger.info(f"[TrustRanker] Enriched {len(enriched)} results with grades.")
        return enriched

    @staticmethod
    def filter_by_grade(enriched_results: List[Dict[str, Any]], min_grade: str = "C") -> List[Dict[str, Any]]:
        """Filter results by minimum grade."""
        grade_order = ["A+", "A", "B", "C", "D", "F"]
        if min_grade not in grade_order:
            min_grade = "C"

        min_index = grade_order.index(min_grade)
        filtered = [r for r in enriched_results if grade_order.index(r.get("grade", "F")) <= min_index]

        logger.info(f"[TrustRanker] Filtered {len(enriched_results)} → {len(filtered)} results (min_grade={min_grade})")
        return filtered
