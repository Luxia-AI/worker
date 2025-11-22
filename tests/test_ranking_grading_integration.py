"""
Integration test: End-to-end retrieval, ranking, and trust grading.
Demonstrates the full pipeline: VDB + KG → Hybrid Rank → Trust Grade.
"""

import pytest

from app.services.ranking.hybrid_ranker import hybrid_rank
from app.services.ranking.trust_ranker import TrustRanker


class TestRankingAndGradingPipeline:
    """Integration tests showing realistic ranking + grading workflows."""

    def test_vdb_and_kg_merge_and_rank(self):
        """Test merging VDB (semantic) and KG (structural) candidates."""
        # Simulate VDB results (cosine similarity scores)
        vdb_results = [
            {
                "score": 0.92,  # High semantic similarity
                "statement": "COVID-19 vaccines have >90% effectiveness against severe disease",
                "entities": ["COVID-19", "vaccines", "effectiveness"],
                "source_url": "https://www.cdc.gov/coronavirus/",
                "published_at": "2024-01-15",
                "credibility": 0.95,
            },
            {
                "score": 0.78,
                "statement": "Vitamin D levels may help with immune function",
                "entities": ["Vitamin D", "immune function"],
                "source_url": "https://health.harvard.edu/",
                "published_at": "2023-11-10",
                "credibility": 0.85,
            },
        ]

        # Simulate KG results (relation confidence scores)
        kg_results = [
            {
                "score": 0.88,  # High path quality
                "statement": "COVID-19 CAUSES respiratory_distress",
                "entities": ["COVID-19", "respiratory_distress"],
                "source_url": "https://www.nih.gov/",
                "published_at": None,
                "credibility": 0.95,
            },
            {
                "score": 0.65,
                "statement": "vaccines PREVENT infection",
                "entities": ["vaccines", "infection"],
                "source_url": "https://www.who.int/",
                "published_at": None,
                "credibility": 0.95,
            },
        ]

        # Hybrid rank
        ranked = hybrid_rank(vdb_results, kg_results, query_entities=["COVID-19", "vaccines"])

        # Assertions
        assert len(ranked) >= 2
        # Top result should have high combined score
        assert ranked[0]["final_score"] > 0.75

    def test_semantic_retrieval_with_metadata(self):
        """Test that VDB retrieval includes all metadata needed for grading."""
        vdb_candidates = [
            {
                "score": 0.89,  # Cosine similarity (will be normalized)
                "statement": "Study shows garlic supplements have minimal health benefit",
                "entities": ["garlic supplements", "health benefit"],
                "source_url": "https://health.harvard.edu/nutrition",
                "published_at": "2024-01-01",
                "credibility": 0.85,
                "source": "Harvard Health",
            }
        ]

        ranked = hybrid_rank(vdb_candidates, [], query_entities=["garlic", "supplements"])
        assert len(ranked) == 1
        assert ranked[0]["statement"] == vdb_candidates[0]["statement"]
        assert ranked[0]["credibility"] == 0.85
        assert "published_at" in ranked[0]

    def test_kg_retrieval_with_path_metrics(self):
        """Test that KG retrieval includes path quality and hop distance."""
        kg_candidates = [
            {
                "score": 0.85,  # Path quality (relation confidence - hop penalty)
                "statement": "Aspirin REDUCES cardiovascular_risk",
                "entities": ["Aspirin", "cardiovascular_risk"],
                "source_url": "https://www.nih.gov/health",
                "credibility": 0.95,
                "hop_distance": 1,
                "confidence": 0.95,  # Raw relation confidence
                "path_quality_score": 0.85,
            }
        ]

        ranked = hybrid_rank([], kg_candidates, query_entities=["Aspirin", "heart"])
        assert len(ranked) == 1
        assert ranked[0]["kg_score"] > 0.0

    def test_full_pipeline_vdb_rank_grade(self):
        """Full pipeline: VDB retrieval → hybrid rank → trust grade."""
        vdb_results = [
            {
                "score": 0.94,
                "statement": "mRNA vaccines produced by Pfizer and Moderna have >95% effectiveness",
                "entities": ["mRNA vaccines", "effectiveness", "Pfizer", "Moderna"],
                "source_url": "https://www.cdc.gov/",
                "published_at": "2024-01-10",
                "credibility": 0.95,
            },
            {
                "score": 0.82,
                "statement": "Some people report mild side effects from COVID vaccines",
                "entities": ["COVID vaccines", "side effects"],
                "source_url": "https://medicalnewstoday.com/",
                "published_at": "2024-01-05",
                "credibility": 0.60,
            },
            {
                "score": 0.45,
                "statement": "Vaccines contain microchips (FALSE)",
                "entities": ["vaccines", "microchips"],
                "source_url": "https://conspiracy-blog.example.com/",
                "published_at": "2024-01-01",
                "credibility": 0.10,
            },
        ]

        # Hybrid rank
        ranked = hybrid_rank(vdb_results, [], query_entities=["vaccines"])

        # Enrich with grades
        graded = TrustRanker.enrich_ranked_results(ranked)

        # Verify grading
        assert len(graded) == 3
        # Top result should be better than bottom (credibility + semantic + entity match)
        assert graded[0]["final_score"] > graded[2]["final_score"]
        # Bottom result should be low grade (conspiracy, low credibility)
        assert graded[2]["grade"] == "F"

    def test_full_pipeline_kg_rank_grade(self):
        """Full pipeline: KG retrieval → hybrid rank → trust grade."""
        kg_results = [
            {
                "score": 0.90,
                "statement": "insulin TREATS diabetes",
                "entities": ["insulin", "diabetes"],
                "source_url": "https://www.nih.gov/",
                "credibility": 0.95,
                "hop_distance": 1,
            },
            {
                "score": 0.75,
                "statement": "exercise IMPROVES_cardiovascular_health",
                "entities": ["exercise", "cardiovascular health"],
                "source_url": "https://health.harvard.edu/",
                "credibility": 0.85,
                "hop_distance": 1,
            },
            {
                "score": 0.40,
                "statement": "homeopathy CURES cancer",
                "entities": ["homeopathy", "cancer"],
                "source_url": "https://questionable-health-site.com/",
                "credibility": 0.30,
                "hop_distance": 2,
            },
        ]

        ranked = hybrid_rank([], kg_results, query_entities=["diabetes", "treatment"])
        graded = TrustRanker.enrich_ranked_results(ranked)

        assert len(graded) == 3
        # Ranking should order by final score, not raw KG score
        assert graded[0]["final_score"] >= graded[1]["final_score"]
        assert graded[1]["final_score"] >= graded[2]["final_score"]
        # Low credibility + low KG score should get poor grade
        assert graded[2]["grade"] == "F"

    def test_mixed_vdb_and_kg_ranking_and_grading(self):
        """Full pipeline: Mix of VDB + KG → hybrid rank → trust grade."""
        vdb_results = [
            {
                "score": 0.91,
                "statement": "Regular exercise improves heart health",
                "entities": ["exercise", "heart health"],
                "source_url": "https://www.cdc.gov/",
                "published_at": "2024-01-01",
                "credibility": 0.95,
            }
        ]

        kg_results = [
            {
                "score": 0.87,
                "statement": "exercise REDUCES_heart_disease_risk",
                "entities": ["exercise", "heart disease risk"],
                "source_url": "https://www.nih.gov/",
                "credibility": 0.95,
                "hop_distance": 1,
            },
            {
                "score": 0.68,
                "statement": "sport IMPROVES fitness",
                "entities": ["sport", "fitness"],
                "source_url": "https://health.harvard.edu/",
                "credibility": 0.85,
                "hop_distance": 2,
            },
        ]

        # Hybrid rank combines both
        ranked = hybrid_rank(vdb_results, kg_results, query_entities=["exercise", "health"])
        assert len(ranked) >= 2

        # Grade all results
        graded = TrustRanker.enrich_ranked_results(ranked)
        assert all("grade" in r for r in graded)
        assert all("grade_rationale" in r for r in graded)

    def test_grade_filtering_workflow(self):
        """Test filtering by grade to identify high-confidence evidence only."""
        # Simulate a mixed quality set of results
        vdb_results = [
            {
                "score": 0.95,
                "statement": "Fact A (excellent)",
                "entities": [],
                "source_url": "https://who.int/",
                "published_at": "2024-01-01",
                "credibility": 0.95,
            },
            {
                "score": 0.75,
                "statement": "Fact B (good)",
                "entities": [],
                "source_url": "https://health.edu/",
                "published_at": "2023-12-01",
                "credibility": 0.80,
            },
            {
                "score": 0.55,
                "statement": "Fact C (poor)",
                "entities": [],
                "source_url": "https://blog.example.com/",
                "published_at": "2023-01-01",
                "credibility": 0.40,
            },
            {
                "score": 0.30,
                "statement": "Fact D (unacceptable)",
                "entities": [],
                "source_url": "https://unknown.site/",
                "published_at": "2022-01-01",
                "credibility": 0.20,
            },
        ]

        ranked = hybrid_rank(vdb_results, [])
        graded = TrustRanker.enrich_ranked_results(ranked)

        # Filter to only A+ and A grades
        high_confidence = TrustRanker.filter_by_grade(graded, min_grade="A")
        assert len(high_confidence) <= len(graded)
        if high_confidence:
            assert all(r["grade"] in ["A+", "A"] for r in high_confidence)

        # Filter to B and above
        good_confidence = TrustRanker.filter_by_grade(graded, min_grade="B")
        assert len(good_confidence) >= len(high_confidence)

    def test_evidence_quality_summary(self):
        """Test generating a summary of evidence quality distribution."""
        vdb_results = [
            {
                "score": 0.93,
                "statement": "Fact 1",
                "entities": ["fact"],
                "source_url": "https://cdc.gov/",
                "published_at": "2024-01-01",
                "credibility": 0.95,
            },
            {
                "score": 0.82,
                "statement": "Fact 2",
                "entities": ["fact"],
                "source_url": "https://news.com/",
                "published_at": "2024-01-01",
                "credibility": 0.60,
            },
            {
                "score": 0.71,
                "statement": "Fact 3",
                "entities": ["fact"],
                "source_url": "https://blog.com/",
                "published_at": "2023-12-01",
                "credibility": 0.45,
            },
            {
                "score": 0.48,
                "statement": "Fact 4",
                "entities": ["fact"],
                "source_url": "https://unknown.com/",
                "published_at": "2023-01-01",
                "credibility": 0.30,
            },
        ]

        ranked = hybrid_rank(vdb_results, [], query_entities=["fact"])
        graded = TrustRanker.enrich_ranked_results(ranked)

        # Get grade distribution
        dist = TrustRanker._grade_distribution(graded)

        # Verify distribution
        assert len(dist) > 0
        # Should have various grades due to different source credibilities
        # High credibility source (CDC) with decent score should get some lift
        assert any(r["final_score"] > 0.5 for r in graded), "At least some results should score > 0.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
