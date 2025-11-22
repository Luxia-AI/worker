"""
Trust-Ranking Module: Maps confidence scores to letter grades (A+ → F).
Provides transparency and interpretability for evidence quality.
"""

from typing import Any, Dict, List

from app.constants.config import (
    GRADE_THRESHOLDS,
    SEMANTIC_THRESHOLD_FAIR,
    SEMANTIC_THRESHOLD_GOOD,
    SEMANTIC_THRESHOLD_HIGH,
    SEMANTIC_THRESHOLD_MIN,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


class TrustRanker:
    """
    Converts hybrid ranking scores (0.0 - 1.0) into interpretable trust grades.

    Grades (A+ → F):
      - A+: final_score >= 0.90 → Excellent evidence (high similarity + credibility + entity match)
      - A:  final_score >= 0.80 → Very good evidence
      - B:  final_score >= 0.70 → Good evidence
      - C:  final_score >= 0.60 → Fair evidence
      - D:  final_score >= 0.50 → Poor evidence
      - F:  final_score <  0.50 → Insufficient evidence (below threshold)

    Grade assignment also considers:
      - Semantic similarity thresholds (cosine similarity with query)
      - Source credibility (authority, gov/edu, news, default)
      - Entity overlap with query
      - Publication recency
    """

    GRADES = ["A+", "A", "B", "C", "D", "F"]

    @staticmethod
    def assign_grade(final_score: float) -> str:
        """
        Map final_score [0.0, 1.0] to a trust grade.

        Args:
            final_score: Hybrid ranking score (normalized [0, 1])

        Returns:
            Grade string: "A+", "A", "B", "C", "D", or "F"
        """
        final_score = max(0.0, min(final_score, 1.0))

        if final_score >= GRADE_THRESHOLDS["A_PLUS"]:
            return "A+"
        elif final_score >= GRADE_THRESHOLDS["A"]:
            return "A"
        elif final_score >= GRADE_THRESHOLDS["B"]:
            return "B"
        elif final_score >= GRADE_THRESHOLDS["C"]:
            return "C"
        elif final_score >= GRADE_THRESHOLDS["D"]:
            return "D"
        else:
            return "F"

    @staticmethod
    def assign_semantic_confidence(semantic_score: float) -> str:
        """
        Assess semantic similarity confidence level.

        Args:
            semantic_score: Normalized semantic similarity [0, 1]

        Returns:
            Confidence level: "HIGH", "GOOD", "FAIR", "LOW", or "NONE"
        """
        semantic_score = max(0.0, min(semantic_score, 1.0))

        if semantic_score >= SEMANTIC_THRESHOLD_HIGH:
            return "HIGH"
        elif semantic_score >= SEMANTIC_THRESHOLD_GOOD:
            return "GOOD"
        elif semantic_score >= SEMANTIC_THRESHOLD_FAIR:
            return "FAIR"
        elif semantic_score >= SEMANTIC_THRESHOLD_MIN:
            return "LOW"
        else:
            return "NONE"

    @staticmethod
    def enrich_ranked_results(ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add trust grades and semantic confidence to ranked results.

        Args:
            ranked_results: List of dicts from hybrid_rank() with keys:
                - final_score: float [0, 1]
                - sem_score: float [0, 1] (semantic similarity)
                - credibility: float [0, 1]
                - entity_overlap: float [0, 1]
                - recency: float [0, 1]
                - (other fields preserved)

        Returns:
            Same list with added fields per item:
                - "grade": str (A+, A, B, C, D, F)
                - "semantic_confidence": str (HIGH, GOOD, FAIR, LOW, NONE)
                - "grade_rationale": str (explanation of grade)
        """
        enriched = []

        for result in ranked_results:
            final_score = result.get("final_score", 0.0)
            sem_score = result.get("sem_score", 0.0)
            credibility = result.get("credibility", 0.5)
            entity_overlap = result.get("entity_overlap", 0.0)
            recency = result.get("recency", 0.0)

            grade = TrustRanker.assign_grade(final_score)
            sem_confidence = TrustRanker.assign_semantic_confidence(sem_score)

            # Build rationale explaining the grade
            factors = []
            if sem_score >= SEMANTIC_THRESHOLD_GOOD:
                factors.append(f"high semantic match ({sem_score:.2f})")
            elif sem_score >= SEMANTIC_THRESHOLD_FAIR:
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

        logger.info(
            f"[TrustRanker] Enriched {len(enriched)} results with grades. "
            f"Distribution: {TrustRanker._grade_distribution(enriched)}"
        )
        return enriched

    @staticmethod
    def _grade_distribution(enriched_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count grade distribution for logging."""
        dist = {g: 0 for g in TrustRanker.GRADES}
        for result in enriched_results:
            grade = result.get("grade", "F")
            dist[grade] = dist.get(grade, 0) + 1
        return {k: v for k, v in dist.items() if v > 0}

    @staticmethod
    def filter_by_grade(enriched_results: List[Dict[str, Any]], min_grade: str = "C") -> List[Dict[str, Any]]:
        """
        Filter results to only include those at or above a minimum grade.

        Args:
            enriched_results: Results from enrich_ranked_results()
            min_grade: Minimum acceptable grade ("A+", "A", "B", "C", "D", "F")

        Returns:
            Filtered list containing only results with grade >= min_grade
        """
        grade_order = ["A+", "A", "B", "C", "D", "F"]
        if min_grade not in grade_order:
            logger.warning(f"[TrustRanker] Invalid min_grade '{min_grade}', using 'C'")
            min_grade = "C"

        min_index = grade_order.index(min_grade)
        filtered = [r for r in enriched_results if grade_order.index(r.get("grade", "F")) <= min_index]

        logger.info(f"[TrustRanker] Filtered {len(enriched_results)} → {len(filtered)} results (min_grade={min_grade})")
        return filtered
