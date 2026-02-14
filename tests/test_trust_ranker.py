"""
Tests for TrustRanker: Grade assignment, semantic confidence, filtering, enrichment.
"""

import pytest

from app.services.ranking.trust_ranker import EvidenceItem, TrustRanker, TrustRankingModule
from app.services.retrieval.kg_normalizer import KGTriple, triple_to_statement, triples_to_evidence


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

    def test_enrich_grade_uses_blended_score_with_high_semantic_and_credibility(self):
        results = [
            {
                "statement": "Calcium is needed to build and maintain strong bones.",
                "final_score": 0.586,
                "sem_score": 0.876,
                "credibility": 0.95,
            }
        ]
        enriched = TrustRanker.enrich_ranked_results(results)
        # Avoid under-grading high-quality evidence due to a conservative final_score.
        assert enriched[0]["grade"] in {"C", "B", "A", "A+"}

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


class TestTrustRankingModule:
    """Test TrustRankingModule functionality."""

    @pytest.fixture
    def module(self):
        return TrustRankingModule()

    def test_normalize_url_for_dedupe_scheme_and_domain(self, module):
        """Test URL normalization: http/https, www. stripping, port stripping."""
        # Different schemes and www. should normalize to same
        url1 = "http://who.int/page"
        url2 = "https://www.who.int/page/"
        url3 = "https://who.int:443/page"
        norm1 = module._normalize_url_for_dedupe(url1)
        norm2 = module._normalize_url_for_dedupe(url2)
        norm3 = module._normalize_url_for_dedupe(url3)
        assert norm1 == norm2 == norm3 == "https://who.int/page"

    def test_normalize_url_keeps_ref_and_source(self, module):
        """Test that 'ref' and 'source' query params are NOT removed."""
        url1 = "https://example.com/page?ref=abc"
        url2 = "https://example.com/page?source=xyz"
        norm1 = module._normalize_url_for_dedupe(url1)
        norm2 = module._normalize_url_for_dedupe(url2)
        assert norm1 == "https://example.com/page?ref=abc"
        assert norm2 == "https://example.com/page?source=xyz"
        assert norm1 != norm2  # Should be distinct

    def test_normalize_url_removes_utm_params(self, module):
        """Test that utm_* params are still removed."""
        url = "https://example.com/page?utm_source=google&utm_medium=email"
        norm = module._normalize_url_for_dedupe(url)
        assert norm == "https://example.com/page"

    def test_normalize_url_sorts_query_params(self, module):
        """Test that query params are sorted deterministically."""
        url1 = "https://example.com/page?z=1&a=2"
        url2 = "https://example.com/page?a=2&z=1"
        norm1 = module._normalize_url_for_dedupe(url1)
        norm2 = module._normalize_url_for_dedupe(url2)
        assert norm1 == norm2 == "https://example.com/page?a=2&z=1"

    def test_statement_hash_normalization(self, module):
        """Test that statement hash dedupes case and whitespace differences."""
        evidence = [
            EvidenceItem("Test statement", 0.8, "", stance="entails"),
            EvidenceItem("  TEST STATEMENT  ", 0.7, "", stance="entails"),
            EvidenceItem("test statement", 0.6, "", stance="entails"),
        ]
        ranked = module.rank_evidence(evidence)
        assert len(ranked) == 1  # Should dedupe to one item

    def test_compute_post_trust_confidence_intervals(self, module):
        """Test confidence intervals in compute_post_trust."""
        evidence = [
            EvidenceItem("Evidence 1", 0.8, "https://example.com/1", stance="entails"),
            EvidenceItem("Evidence 2", 0.6, "https://example.com/2", stance="entails"),
            EvidenceItem("Evidence 3", 0.4, "https://example.com/3", stance="neutral"),
        ]
        ranked = module.rank_evidence(evidence)
        result = module.compute_post_trust(ranked, top_k=10)

        # Check new keys exist
        assert "trust_post_ci_low" in result
        assert "trust_post_ci_high" in result
        assert "trust_post_ci_method" in result
        assert "trust_post_ci_samples" in result
        assert "trust_post_ci_level" in result

        # Check values
        assert result["trust_post_ci_method"] == "bootstrap"
        assert result["trust_post_ci_samples"] == 200
        assert result["trust_post_ci_level"] == 0.95
        assert result["trust_post_ci_low"] <= result["trust_post"] <= result["trust_post_ci_high"]

    def test_compute_post_trust_ci_deterministic(self, module):
        """Test that confidence intervals are deterministic."""
        evidence = [
            EvidenceItem("Evidence 1", 0.8, "https://example.com/1", stance="entails"),
            EvidenceItem("Evidence 2", 0.6, "https://example.com/2", stance="entails"),
        ]
        ranked = module.rank_evidence(evidence)
        result1 = module.compute_post_trust(ranked, top_k=10)
        result2 = module.compute_post_trust(ranked, top_k=10)

        assert result1["trust_post_ci_low"] == result2["trust_post_ci_low"]
        assert result1["trust_post_ci_high"] == result2["trust_post_ci_high"]

    def test_compute_post_trust_ci_edge_cases(self, module):
        """Test CI for empty and single evidence."""
        # Empty
        result_empty = module.compute_post_trust([], top_k=10)
        assert result_empty["trust_post"] == 0.0
        assert result_empty["trust_post_ci_low"] == 0.0
        assert result_empty["trust_post_ci_high"] == 0.0

        # Single evidence
        evidence = [EvidenceItem("Single", 0.7, "https://example.com", stance="entails")]
        ranked = module.rank_evidence(evidence)
        result_single = module.compute_post_trust(ranked, top_k=10)
        assert result_single["trust_post_ci_low"] == result_single["trust_post"]
        assert result_single["trust_post_ci_high"] == result_single["trust_post"]

    def test_score_components_set_in_rank_evidence(self, module):
        """Test that score_components are set for each evidence item after ranking."""
        evidence = [
            EvidenceItem("Evidence 1", 0.8, "https://who.int/page1", stance="entails"),
            EvidenceItem("Evidence 2", 0.6, "https://cdc.gov/page2", stance="neutral"),
        ]
        ranked = module.rank_evidence(evidence)
        for item in ranked:
            assert item.score_components is not None
            assert "semantic" in item.score_components
            assert "source" in item.score_components
            assert "recency" in item.score_components
            assert "stance_raw" in item.score_components
            assert "stance_mapped" in item.score_components
            assert "trust" in item.score_components
            assert item.score_components["trust"] == item.trust

    def test_post_breakdown_in_compute_post_trust(self, module):
        """Test that post_breakdown is included with expected keys."""
        evidence = [
            EvidenceItem("Evidence 1", 0.8, "https://who.int/page1", stance="entails"),
            EvidenceItem("Evidence 2", 0.6, "https://cdc.gov/page2", stance="neutral"),
            EvidenceItem("Evidence 3", 0.4, "https://example.com/page3", stance="contradicts"),
        ]
        ranked = module.rank_evidence(evidence)
        result = module.compute_post_trust(ranked, top_k=10)

        assert "post_breakdown" in result
        breakdown = result["post_breakdown"]
        expected_keys = [
            "semantic_mean",
            "source_mean",
            "recency_mean",
            "stance_mapped_mean",
            "evidence_used",
            "entails_count",
            "contradicts_count",
            "neutral_count",
            "top_sources",
        ]
        for key in expected_keys:
            assert key in breakdown

        assert breakdown["evidence_used"] == 3
        assert breakdown["entails_count"] == 1
        assert breakdown["contradicts_count"] == 1
        assert breakdown["neutral_count"] == 1
        assert isinstance(breakdown["top_sources"], list)


class TestKGNormalizer:
    """Test KG triple normalization."""

    def test_triple_to_statement_basic(self):
        """Test basic triple to statement conversion."""
        triple = KGTriple(subject="COVID-19", relation="causes", object="fever")
        statement = triple_to_statement(triple)
        assert statement == "COVID-19 causes fever."

    def test_triple_to_statement_underscores(self):
        """Test relation with underscores gets spaces."""
        triple = KGTriple(subject="Vaccine", relation="prevents_disease", object="infection")
        statement = triple_to_statement(triple)
        assert statement == "Vaccine prevents disease infection."

    def test_triple_to_statement_whitespace(self):
        """Test whitespace normalization."""
        triple = KGTriple(subject="  COVID-19  ", relation="  causes  ", object="  fever  ")
        statement = triple_to_statement(triple)
        assert statement == "COVID-19 causes fever."

    def test_triples_to_evidence(self):
        """Test conversion of triples to EvidenceItem list."""
        triples = [
            KGTriple(subject="COVID-19", relation="causes", object="fever", source_url="https://who.int"),
            KGTriple(subject="Vaccine", relation="prevents", object="COVID-19", published_at="2023-01-01"),
        ]

        def semantic_score_provider(statement: str) -> float:
            return 0.8  # Mock score

        evidence = triples_to_evidence(triples, semantic_score_provider)

        assert len(evidence) == 2
        assert evidence[0].statement == "COVID-19 causes fever."
        assert evidence[0].semantic_score == 0.8
        assert evidence[0].source_url == "https://who.int"
        assert evidence[0].stance == "neutral"
        assert evidence[1].statement == "Vaccine prevents COVID-19."
        assert evidence[1].published_at == "2023-01-01"


class TestNeo4jKGRepository:
    """Test Neo4j KG repository functionality."""

    @pytest.mark.asyncio
    async def test_fetch_triples_mapping(self):
        """Test that Neo4j records are correctly mapped to KGTriple objects."""
        from unittest.mock import MagicMock, patch

        from app.services.retrieval.neo4j_kg_repository import Neo4jKGRepository

        # Mock Neo4j client and session
        mock_client = MagicMock()

        # Mock successful query execution
        mock_record1 = MagicMock()
        mock_record1.get.side_effect = lambda key: {
            "subject": "COVID-19",
            "subject_id": "5b7f9fa40bdc1207",
            "predicate": "causes",
            "rid": "5b7f9fa40bdc1207|causes|abc123def4567890",
            "object": "fever",
            "object_id": "abc123def4567890",
            "source_url": "https://who.int",
            "source_domain": "who.int",
            "confidence": 0.9,
            "updated_at": "2023-01-01T00:00:00Z",
        }.get(key)

        mock_record2 = MagicMock()
        mock_record2.get.side_effect = lambda key: {
            "subject": "Vaccine",
            "subject_id": "d4aac1a7c59ad68f",
            "predicate": "prevents",
            "rid": "d4aac1a7c59ad68f|prevents|5b7f9fa40bdc1207",
            "object": "COVID-19",
            "object_id": "5b7f9fa40bdc1207",
            "source_url": None,  # Test null handling
            "source_domain": None,
            "confidence": 0.0,  # coalesce default
            "updated_at": None,
        }.get(key)

        # Mock the retry function to return records directly
        with patch("app.services.retrieval.neo4j_kg_repository._execute_with_retry") as mock_retry:
            mock_retry.return_value = [mock_record1, mock_record2]

            repo = Neo4jKGRepository(mock_client)
            triples = await repo.fetch_triples_for_claim(entity_ids=["5b7f9fa40bdc1207", "d4aac1a7c59ad68f"], limit=10)

            # Verify retry was called with correct parameters
            mock_retry.assert_called_once()
            call_args = mock_retry.call_args
            assert "fetch_kg_triples_for_claim" in call_args.kwargs["query_name"]
            assert call_args.kwargs["params"] == {"entity_ids": ["5b7f9fa40bdc1207", "d4aac1a7c59ad68f"], "limit": 10}

            # Verify mapping
            assert len(triples) == 2
            assert triples[0].subject == "COVID-19"
            assert triples[0].subject_id == "5b7f9fa40bdc1207"
            assert triples[0].relation == "causes"
            assert triples[0].rid == "5b7f9fa40bdc1207|causes|abc123def4567890"
            assert triples[0].object == "fever"
            assert triples[0].object_id == "abc123def4567890"
            assert triples[0].source_url == "https://who.int"
            assert triples[0].source_domain == "who.int"
            assert triples[0].confidence == 0.9
            assert triples[0].published_at == "2023-01-01T00:00:00Z"

            assert triples[1].subject == "Vaccine"
            assert triples[1].subject_id == "d4aac1a7c59ad68f"
            assert triples[1].relation == "prevents"
            assert triples[1].rid == "d4aac1a7c59ad68f|prevents|5b7f9fa40bdc1207"
            assert triples[1].object == "COVID-19"
            assert triples[1].object_id == "5b7f9fa40bdc1207"
            assert triples[1].source_url is None
            assert triples[1].source_domain is None
            assert triples[1].confidence == 0.0  # coalesce default
            assert triples[1].published_at is None

    @pytest.mark.asyncio
    async def test_fetch_triples_with_claim_id(self):
        """Test fetching triples using claim_id parameter."""
        from unittest.mock import MagicMock, patch

        from app.services.retrieval.neo4j_kg_repository import Neo4jKGRepository

        mock_client = MagicMock()

        mock_record = MagicMock()
        mock_record.get.side_effect = lambda key: {
            "subject": "COVID-19",
            "subject_id": "5b7f9fa40bdc1207",
            "predicate": "causes",
            "rid": "5b7f9fa40bdc1207|causes|abc123def4567890",
            "object": "fever",
            "object_id": "abc123def4567890",
            "source_url": "https://who.int",
            "source_domain": "who.int",
            "confidence": 0.8,
            "updated_at": "2023-01-01T00:00:00Z",
        }.get(key)

        with patch("app.services.retrieval.neo4j_kg_repository._execute_with_retry") as mock_retry:
            mock_retry.return_value = [mock_record]

            repo = Neo4jKGRepository(mock_client)
            triples = await repo.fetch_triples_for_claim(claim_id="claim-123", limit=5)

            mock_retry.assert_called_once()
            call_args = mock_retry.call_args
            assert call_args.kwargs["params"] == {"claim_id": "claim-123", "limit": 5}
            assert len(triples) == 1

    @pytest.mark.asyncio
    async def test_fetch_triples_validation_error(self):
        """Test that method raises error when neither claim_id nor entity_ids provided."""
        from app.services.retrieval.neo4j_kg_repository import Neo4jKGRepository

        repo = Neo4jKGRepository()

        with pytest.raises(ValueError, match="Either claim_id or entity_ids must be provided"):
            await repo.fetch_triples_for_claim()

    @pytest.mark.asyncio
    async def test_fetch_triples_limit_enforcement(self):
        """Test that limits are properly enforced and truncated if needed."""
        from unittest.mock import MagicMock, patch

        from app.services.retrieval.neo4j_kg_repository import Neo4jKGRepository

        mock_client = MagicMock()

        # Mock records that exceed limit
        mock_records = [MagicMock()] * 150  # More than limit of 100

        with patch("app.services.retrieval.neo4j_kg_repository._execute_with_retry") as mock_retry:
            mock_retry.return_value = mock_records

            repo = Neo4jKGRepository(mock_client)
            triples = await repo.fetch_triples_for_claim(entity_ids=["test-entity-id"], limit=100)

            # Should truncate to limit
            assert len(triples) <= 100

    @pytest.mark.asyncio
    async def test_store_evidence_metadata_success(self):
        """Test successful storage of evidence metadata."""
        from unittest.mock import MagicMock, patch

        from app.services.retrieval.neo4j_kg_repository import Neo4jKGRepository

        mock_client = MagicMock()

        with patch("app.services.retrieval.neo4j_kg_repository._execute_with_retry") as mock_retry:
            mock_retry.return_value = []  # No results expected for write

            repo = Neo4jKGRepository(mock_client)
            success = await repo.store_evidence_metadata(
                claim_text="Test claim",
                evidence_count=5,
                sources_used=["https://example.com"],
                processing_timestamp="2024-01-01T00:00:00Z",
            )

            assert success is True
            mock_retry.assert_called_once()
            call_args = mock_retry.call_args
            assert "store_evidence_metadata" in call_args.kwargs["query_name"]
            params = call_args.kwargs["params"]
            assert "claim_hash" in params
            assert params["evidence_count"] == 5
            assert params["sources_used"] == ["https://example.com"]

    @pytest.mark.asyncio
    async def test_store_evidence_metadata_failure(self):
        """Test failure handling in evidence metadata storage."""
        from unittest.mock import MagicMock, patch

        from app.services.retrieval.neo4j_kg_repository import Neo4jKGRepository

        mock_client = MagicMock()

        with patch("app.services.retrieval.neo4j_kg_repository._execute_with_retry") as mock_retry:
            mock_retry.side_effect = Exception("Connection failed")

            repo = Neo4jKGRepository(mock_client)
            success = await repo.store_evidence_metadata(
                claim_text="Test claim",
                evidence_count=5,
                sources_used=["https://example.com"],
                processing_timestamp="2024-01-01T00:00:00Z",
            )

            assert success is False

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        from unittest.mock import AsyncMock, patch

        from app.services.retrieval.neo4j_kg_repository import _execute_with_retry

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [{"test": "data"}]
        mock_session.run.return_value = mock_result

        with patch("app.services.retrieval.neo4j_kg_repository.logger") as mock_logger:
            records = await _execute_with_retry(
                query="RETURN 1", params={}, session=mock_session, query_name="test_query"
            )

            assert records == [{"test": "data"}]
            # Verify success logging
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args
            assert "test_query" in call_args[0][0]
            assert call_args[1]["extra"]["success"] is True
            assert call_args[1]["extra"]["retry_count"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_retry_transient_error_retry(self):
        """Test retry logic for transient errors."""
        from unittest.mock import AsyncMock, patch

        from app.services.retrieval.neo4j_kg_repository import _execute_with_retry

        mock_session = AsyncMock()

        # First two calls fail with transient error, third succeeds
        mock_session.run.side_effect = [
            Exception("Neo.TransientError.Network.CommunicationError"),
            Exception("Neo.TransientError.General.DatabaseUnavailable"),
            AsyncMock(data=AsyncMock(return_value=[{"success": True}])),
        ]

        with patch("app.services.retrieval.neo4j_kg_repository.logger") as mock_logger, patch("asyncio.sleep"):

            records = await _execute_with_retry(
                query="RETURN 1", params={}, session=mock_session, query_name="test_query"
            )

            assert records == [{"success": True}]
            # Should have logged warnings for retries
            warning_calls = [call for call in mock_logger.warning.call_args_list if "failed" in str(call)]
            assert len(warning_calls) == 2  # Two failures before success
            # Should have logged backoff delays
            info_calls = [call for call in mock_logger.info.call_args_list if "Retrying" in str(call)]
            assert len(info_calls) == 2  # Two retry attempts

    @pytest.mark.asyncio
    async def test_execute_with_retry_permanent_error_no_retry(self):
        """Test that permanent errors don't trigger retry."""
        from unittest.mock import AsyncMock, patch

        from app.services.retrieval.neo4j_kg_repository import _execute_with_retry

        mock_session = AsyncMock()
        mock_session.run.side_effect = Exception("Syntax error in Cypher query")

        with (
            patch("app.services.retrieval.neo4j_kg_repository.logger") as mock_logger,
            patch("asyncio.sleep") as mock_sleep,
        ):

            with pytest.raises(Exception, match="Syntax error"):
                await _execute_with_retry(
                    query="INVALID QUERY", params={}, session=mock_session, query_name="test_query"
                )

            # Should log once and not retry
            mock_logger.warning.assert_called_once()
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_with_retry_max_retries_exhausted(self):
        """Test behavior when all retries are exhausted."""
        from unittest.mock import AsyncMock, patch

        from app.services.retrieval.neo4j_kg_repository import _execute_with_retry

        mock_session = AsyncMock()
        # Always fail with transient error
        mock_session.run.side_effect = Exception("Neo.TransientError.Network.CommunicationError")

        with patch("app.services.retrieval.neo4j_kg_repository.logger") as mock_logger, patch("asyncio.sleep"):

            with pytest.raises(Exception):
                await _execute_with_retry(query="RETURN 1", params={}, session=mock_session, query_name="test_query")

            # Should have 4 attempts (initial + 3 retries)
            assert mock_session.run.call_count == 4
            # Should have logged final error
            mock_logger.error.assert_called_once()


class TestKGIntegration:
    """Test KG integration with trust ranking pipeline."""

    def test_kg_triples_to_evidence_pipeline(self, module):
        """Test that KG triples flow through to ranked evidence."""
        triples = [
            KGTriple("COVID-19", "causes", "fever", "https://who.int", "2023-01-01"),
            KGTriple("Vaccine", "prevents", "COVID-19", "https://cdc.gov", "2023-02-01"),
        ]

        def score_provider(s):
            return 0.8

        evidence = triples_to_evidence(triples, score_provider)

        # Should produce EvidenceItem list
        assert len(evidence) == 2
        assert all(isinstance(item, EvidenceItem) for item in evidence)
        assert evidence[0].statement == "COVID-19 causes fever."
        assert evidence[0].source_url == "https://who.int"
        assert evidence[0].published_at == "2023-01-01"
        assert evidence[0].stance == "neutral"

        # Should rank without errors
        ranked = module.rank_evidence(evidence)
        assert len(ranked) == 2
        assert all(hasattr(item, "score_components") for item in ranked)

    def test_kg_evidence_dedupe_with_url(self, module):
        """Test deduplication of KG evidence using URL normalization."""
        triples = [
            KGTriple("COVID-19", "causes", "fever", "https://who.int/page", "2023-01-01"),
            KGTriple("COVID-19", "causes", "fever", "https://www.who.int/page/", "2023-01-01"),  # Same normalized URL
        ]

        def score_provider(s):
            return 0.8

        evidence = triples_to_evidence(triples, score_provider)

        ranked = module.rank_evidence(evidence)
        assert len(ranked) == 1  # Should dedupe


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
