"""
Tests for TrustRanker: Grade assignment, semantic confidence, filtering, enrichment.
"""

import pytest

from app.services.ranking.trust_ranker import TrustRanker


class TestGradeAssignment:
    """Test letter grade assignment based on final_score."""

    def test_assign_grade_a_plus(self):
        """Test A+ grade for scores >= 0.90."""
        assert TrustRanker.assign_grade(0.90) == "A+"
        assert TrustRanker.assign_grade(0.95) == "A+"
        assert TrustRanker.assign_grade(1.0) == "A+"

    def test_assign_grade_a(self):
        """Test A grade for scores >= 0.80 and < 0.90."""
        assert TrustRanker.assign_grade(0.80) == "A"
        assert TrustRanker.assign_grade(0.85) == "A"
        assert TrustRanker.assign_grade(0.89) == "A"

    def test_assign_grade_b(self):
        """Test B grade for scores >= 0.70 and < 0.80."""
        assert TrustRanker.assign_grade(0.70) == "B"
        assert TrustRanker.assign_grade(0.75) == "B"
        assert TrustRanker.assign_grade(0.79) == "B"

    def test_assign_grade_c(self):
        """Test C grade for scores >= 0.60 and < 0.70."""
        assert TrustRanker.assign_grade(0.60) == "C"
        assert TrustRanker.assign_grade(0.65) == "C"
        assert TrustRanker.assign_grade(0.69) == "C"

    def test_assign_grade_d(self):
        """Test D grade for scores >= 0.50 and < 0.60."""
        assert TrustRanker.assign_grade(0.50) == "D"
        assert TrustRanker.assign_grade(0.55) == "D"
        assert TrustRanker.assign_grade(0.59) == "D"

    def test_assign_grade_f(self):
        """Test F grade for scores < 0.50."""
        assert TrustRanker.assign_grade(0.49) == "F"
        assert TrustRanker.assign_grade(0.25) == "F"
        assert TrustRanker.assign_grade(0.0) == "F"

    def test_assign_grade_out_of_bounds(self):
        """Test that out-of-bounds scores are clamped."""
        # Negative values clamped to 0.0
        assert TrustRanker.assign_grade(-0.5) == "F"
        # Values > 1.0 clamped to 1.0
        assert TrustRanker.assign_grade(1.5) == "A+"


class TestSemanticConfidence:
    """Test semantic similarity confidence level assignment."""

    def test_semantic_confidence_high(self):
        """Test HIGH confidence for semantic_score >= 0.90."""
        assert TrustRanker.assign_semantic_confidence(0.90) == "HIGH"
        assert TrustRanker.assign_semantic_confidence(0.95) == "HIGH"
        assert TrustRanker.assign_semantic_confidence(1.0) == "HIGH"

    def test_semantic_confidence_good(self):
        """Test GOOD confidence for semantic_score >= 0.75 and < 0.90."""
        assert TrustRanker.assign_semantic_confidence(0.75) == "GOOD"
        assert TrustRanker.assign_semantic_confidence(0.82) == "GOOD"
        assert TrustRanker.assign_semantic_confidence(0.89) == "GOOD"

    def test_semantic_confidence_fair(self):
        """Test FAIR confidence for semantic_score >= 0.60 and < 0.75."""
        assert TrustRanker.assign_semantic_confidence(0.60) == "FAIR"
        assert TrustRanker.assign_semantic_confidence(0.70) == "FAIR"
        assert TrustRanker.assign_semantic_confidence(0.74) == "FAIR"

    def test_semantic_confidence_low(self):
        """Test LOW confidence for semantic_score >= 0.40 and < 0.60."""
        assert TrustRanker.assign_semantic_confidence(0.40) == "LOW"
        assert TrustRanker.assign_semantic_confidence(0.50) == "LOW"
        assert TrustRanker.assign_semantic_confidence(0.59) == "LOW"

    def test_semantic_confidence_none(self):
        """Test NONE confidence for semantic_score < 0.40."""
        assert TrustRanker.assign_semantic_confidence(0.39) == "NONE"
        assert TrustRanker.assign_semantic_confidence(0.20) == "NONE"
        assert TrustRanker.assign_semantic_confidence(0.0) == "NONE"


class TestEnrichRankedResults:
    """Test enrichment of ranked results with grades and rationales."""

    def test_enrich_single_result_high_score(self):
        """Test enrichment of a high-scoring result."""
        results = [
            {
                "statement": "Vitamin C boosts immunity",
                "final_score": 0.92,
                "sem_score": 0.88,
                "credibility": 0.95,
                "entity_overlap": 0.8,
                "recency": 0.7,
            }
        ]
        enriched = TrustRanker.enrich_ranked_results(results)

        assert len(enriched) == 1
        assert enriched[0]["grade"] == "A+"
        assert enriched[0]["semantic_confidence"] == "GOOD"
        assert "high semantic match" in enriched[0]["grade_rationale"]
        assert "trusted source" in enriched[0]["grade_rationale"]

    def test_enrich_multiple_results_varied_scores(self):
        """Test enrichment of multiple results with different scores."""
        results = [
            {
                "statement": "Fact 1",
                "final_score": 0.95,
                "sem_score": 0.90,
                "credibility": 0.95,
                "entity_overlap": 0.8,
                "recency": 0.6,
            },
            {
                "statement": "Fact 2",
                "final_score": 0.75,
                "sem_score": 0.70,
                "credibility": 0.70,
                "entity_overlap": 0.5,
                "recency": 0.4,
            },
            {
                "statement": "Fact 3",
                "final_score": 0.45,
                "sem_score": 0.30,
                "credibility": 0.50,
                "entity_overlap": 0.1,
                "recency": 0.2,
            },
        ]
        enriched = TrustRanker.enrich_ranked_results(results)

        assert enriched[0]["grade"] == "A+"
        assert enriched[1]["grade"] == "B"
        assert enriched[2]["grade"] == "F"

    def test_enrich_preserves_original_fields(self):
        """Test that enrichment preserves original result fields."""
        original_result = {
            "statement": "Test fact",
            "final_score": 0.85,
            "sem_score": 0.75,
            "credibility": 0.80,
            "entity_overlap": 0.6,
            "recency": 0.5,
            "source_url": "https://example.com",
            "entities": ["entity1", "entity2"],
        }
        enriched = TrustRanker.enrich_ranked_results([original_result])

        assert enriched[0]["statement"] == original_result["statement"]
        assert enriched[0]["source_url"] == original_result["source_url"]
        assert enriched[0]["entities"] == original_result["entities"]
        assert enriched[0]["final_score"] == original_result["final_score"]

    def test_enrich_missing_fields_defaults(self):
        """Test that enrichment handles missing optional fields."""
        results = [
            {
                "statement": "Minimal fact",
                "final_score": 0.70,
                # Missing sem_score, credibility, entity_overlap, recency
            }
        ]
        enriched = TrustRanker.enrich_ranked_results(results)

        assert enriched[0]["grade"] == "B"
        assert enriched[0]["semantic_confidence"] == "NONE"
        # Should include some rationale even with missing fields
        assert "grade_rationale" in enriched[0]

    def test_enrich_grade_rationale_high_similarity(self):
        """Test rationale generation for high similarity match."""
        results = [
            {
                "statement": "Fact",
                "final_score": 0.92,
                "sem_score": 0.92,
                "credibility": 0.85,
                "entity_overlap": 0.75,
                "recency": 0.6,
            }
        ]
        enriched = TrustRanker.enrich_ranked_results(results)

        rationale = enriched[0]["grade_rationale"]
        assert "high semantic match" in rationale
        assert "0.92" in rationale

    def test_enrich_grade_rationale_low_confidence(self):
        """Test rationale generation for low confidence result."""
        results = [
            {
                "statement": "Fact",
                "final_score": 0.35,
                "sem_score": 0.20,
                "credibility": 0.30,
                "entity_overlap": 0.05,
                "recency": 0.10,
            }
        ]
        enriched = TrustRanker.enrich_ranked_results(results)

        rationale = enriched[0]["grade_rationale"]
        assert "low confidence" in rationale.lower()


class TestFilterByGrade:
    """Test grade-based filtering of results."""

    def test_filter_by_grade_c_minimum(self):
        """Test filtering to minimum grade C."""
        results = [
            {"statement": "A+ fact", "grade": "A+", "final_score": 0.95},
            {"statement": "B fact", "grade": "B", "final_score": 0.75},
            {"statement": "C fact", "grade": "C", "final_score": 0.65},
            {"statement": "D fact", "grade": "D", "final_score": 0.55},
            {"statement": "F fact", "grade": "F", "final_score": 0.30},
        ]
        filtered = TrustRanker.filter_by_grade(results, min_grade="C")

        assert len(filtered) == 3  # A+, B, C grades (no D, F)
        assert all(r["grade"] in ["A+", "A", "B", "C"] for r in filtered)

    def test_filter_by_grade_a_minimum(self):
        """Test filtering to minimum grade A."""
        results = [
            {"statement": "A+ fact", "grade": "A+", "final_score": 0.95},
            {"statement": "A fact", "grade": "A", "final_score": 0.82},
            {"statement": "B fact", "grade": "B", "final_score": 0.75},
            {"statement": "C fact", "grade": "C", "final_score": 0.65},
        ]
        filtered = TrustRanker.filter_by_grade(results, min_grade="A")

        assert len(filtered) == 2  # Only A+ and A
        assert all(r["grade"] in ["A+", "A"] for r in filtered)

    def test_filter_by_grade_f_minimum(self):
        """Test filtering to minimum grade F (all results pass)."""
        results = [
            {"statement": "A fact", "grade": "A", "final_score": 0.85},
            {"statement": "F fact", "grade": "F", "final_score": 0.30},
        ]
        filtered = TrustRanker.filter_by_grade(results, min_grade="F")

        assert len(filtered) == 2  # All pass F threshold

    def test_filter_by_grade_empty_list(self):
        """Test filtering empty result list."""
        filtered = TrustRanker.filter_by_grade([], min_grade="B")
        assert len(filtered) == 0

    def test_filter_by_grade_no_matching_grades(self):
        """Test filtering when no results meet minimum grade."""
        results = [
            {"statement": "F fact", "grade": "F", "final_score": 0.30},
            {"statement": "D fact", "grade": "D", "final_score": 0.55},
        ]
        filtered = TrustRanker.filter_by_grade(results, min_grade="A")

        assert len(filtered) == 0

    def test_filter_by_grade_invalid_grade_uses_default(self):
        """Test that invalid min_grade defaults to C."""
        results = [
            {"statement": "B fact", "grade": "B", "final_score": 0.75},
            {"statement": "D fact", "grade": "D", "final_score": 0.55},
        ]
        # Use invalid grade, should default to C
        filtered = TrustRanker.filter_by_grade(results, min_grade="Z")

        # Should behave as if min_grade="C"
        assert len(filtered) == 1  # Only B passes (C threshold)


class TestGradeDistribution:
    """Test grade distribution counting."""

    def test_grade_distribution_mixed(self):
        """Test distribution counting with mixed grades."""
        results = [
            {"grade": "A+"},
            {"grade": "A+"},
            {"grade": "A"},
            {"grade": "B"},
            {"grade": "C"},
            {"grade": "F"},
        ]
        dist = TrustRanker._grade_distribution(results)

        assert dist["A+"] == 2
        assert dist["A"] == 1
        assert dist["B"] == 1
        assert dist["C"] == 1
        assert dist["F"] == 1
        # No D grades (not present, so not in output)
        assert "D" not in dist

    def test_grade_distribution_empty(self):
        """Test distribution counting with empty list."""
        dist = TrustRanker._grade_distribution([])
        assert len(dist) == 0

    def test_grade_distribution_single_grade(self):
        """Test distribution with only one grade present."""
        results = [
            {"grade": "B"},
            {"grade": "B"},
            {"grade": "B"},
        ]
        dist = TrustRanker._grade_distribution(results)

        assert dist == {"B": 3}


class TestTrustRankerIntegration:
    """Integration tests combining multiple TrustRanker methods."""

    def test_full_pipeline_enrich_and_filter(self):
        """Test full pipeline: enrich results, then filter by grade."""
        # Start with hybrid-ranked results (no grades)
        raw_results = [
            {
                "statement": "Excellent evidence",
                "final_score": 0.92,
                "sem_score": 0.88,
                "credibility": 0.95,
                "entity_overlap": 0.75,
                "recency": 0.65,
            },
            {
                "statement": "Good evidence",
                "final_score": 0.78,
                "sem_score": 0.72,
                "credibility": 0.70,
                "entity_overlap": 0.50,
                "recency": 0.40,
            },
            {
                "statement": "Poor evidence",
                "final_score": 0.48,
                "sem_score": 0.35,
                "credibility": 0.40,
                "entity_overlap": 0.10,
                "recency": 0.20,
            },
        ]

        # Step 1: Enrich with grades
        enriched = TrustRanker.enrich_ranked_results(raw_results)
        assert len(enriched) == 3
        assert enriched[0]["grade"] == "A+"
        assert enriched[1]["grade"] == "B"
        assert enriched[2]["grade"] == "F"

        # Step 2: Filter to only good evidence
        filtered = TrustRanker.filter_by_grade(enriched, min_grade="B")
        assert len(filtered) == 2
        assert all(r["grade"] in ["A+", "A", "B"] for r in filtered)

    def test_end_to_end_ranking_and_grading(self):
        """Test end-to-end: score → grade → rationale."""
        test_cases = [
            (0.95, "A+", "HIGH"),
            (0.85, "A", "GOOD"),
            (0.72, "B", "FAIR"),
            (0.62, "C", "FAIR"),
            (0.52, "D", "LOW"),
            (0.30, "F", "NONE"),
        ]

        for score, expected_grade, expected_confidence in test_cases:
            grade = TrustRanker.assign_grade(score)
            confidence = TrustRanker.assign_semantic_confidence(score)

            assert grade == expected_grade, f"Score {score} → grade {grade} != {expected_grade}"
            assert (
                confidence == expected_confidence
            ), f"Score {score} → confidence {confidence} != {expected_confidence}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
