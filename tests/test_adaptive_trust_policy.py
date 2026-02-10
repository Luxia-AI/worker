"""
Tests for Adaptive Trust Policy Module.

Tests claim decomposition, metric calculation, gating rules, and adaptive trust scoring.
"""

import pytest

from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy
from app.services.ranking.trust_ranker import EvidenceItem


class TestAdaptiveTrustPolicy:
    """Test suite for adaptive trust policy."""

    @pytest.fixture
    def policy(self):
        """Create adaptive trust policy instance."""
        return AdaptiveTrustPolicy()

    @pytest.fixture
    def sample_evidence(self):
        """Create sample evidence items for testing."""
        return [
            EvidenceItem(
                statement="COVID-19 vaccines are effective and safe.",
                semantic_score=0.9,
                source_url="https://who.int/covid-vaccines",
                published_at="2023-01-01",
                trust=0.85,
                stance="entails",
            ),
            EvidenceItem(
                statement="Vaccines reduce hospitalization rates significantly.",
                semantic_score=0.8,
                source_url="https://cdc.gov/vaccine-data",
                published_at="2023-02-01",
                trust=0.80,
                stance="entails",
            ),
            EvidenceItem(
                statement="Some vaccines have rare side effects.",
                semantic_score=0.7,
                source_url="https://nih.gov/vaccine-safety",
                published_at="2023-03-01",
                trust=0.75,
                stance="neutral",
            ),
        ]

    def test_decompose_simple_claim(self, policy):
        """Test decomposition of simple single-part claim."""
        claim = "COVID-19 vaccines are effective."
        subclaims = policy.decompose_claim(claim)

        assert len(subclaims) == 1
        assert subclaims[0].rstrip(".") == claim.rstrip(".")

    def test_decompose_compound_claim(self, policy):
        """Test decomposition of compound claim with conjunction."""
        claim = "COVID-19 vaccines are effective and they reduce hospitalization rates."
        subclaims = policy.decompose_claim(claim)

        assert len(subclaims) == 2
        assert "COVID-19 vaccines are effective" in subclaims
        assert "and they reduce hospitalization rates" in subclaims

    def test_decompose_multiple_sentences(self, policy):
        """Test decomposition of multi-sentence claim."""
        claim = "COVID-19 vaccines are effective. They also reduce hospitalization rates. Side effects are rare."
        subclaims = policy.decompose_claim(claim)

        assert len(subclaims) == 3
        assert any(s.rstrip(".") == "COVID-19 vaccines are effective" for s in subclaims)
        assert any(s.rstrip(".") == "They also reduce hospitalization rates" for s in subclaims)
        assert any(s.rstrip(".") == "Side effects are rare" for s in subclaims)

    def test_decompose_comparative_claim(self, policy):
        """Test decomposition of comparative claim."""
        claim = "COVID-19 vaccines are more effective than natural immunity alone."
        subclaims = policy.decompose_claim(claim)

        assert len(subclaims) == 2
        assert "COVID-19 vaccines are" in subclaims[0]
        assert "more effective than natural immunity alone" in subclaims[1]

    def test_calculate_coverage_full(self, policy, sample_evidence):
        """Test coverage calculation with full coverage."""
        subclaims = [
            "COVID-19 vaccines are effective",
            "vaccines reduce hospitalization rates",
            "vaccines have side effects",
        ]

        coverage = policy.calculate_coverage(subclaims, sample_evidence)
        assert coverage == 1.0  # All subclaims covered

    def test_calculate_coverage_partial(self, policy, sample_evidence):
        """Test coverage calculation with partial coverage."""
        subclaims = [
            "COVID-19 vaccines are effective",
            "vaccines reduce hospitalization rates",
            "sun exposure increases vitamin D naturally",  # Not covered
        ]

        coverage = policy.calculate_coverage(subclaims, sample_evidence)
        assert coverage == pytest.approx(2 / 3, rel=0.1)  # 2 out of 3 covered

    def test_calculate_diversity_single_source(self, policy):
        """Test diversity calculation with single source."""
        evidence = [
            EvidenceItem("test", 0.8, "https://who.int/test", trust=0.8),
            EvidenceItem("test2", 0.7, "https://who.int/test2", trust=0.7),
        ]

        diversity = policy.calculate_diversity(evidence)
        assert diversity < 0.5  # Low diversity with single domain

    def test_calculate_diversity_multiple_sources(self, policy):
        """Test diversity calculation with multiple sources."""
        evidence = [
            EvidenceItem("test", 0.8, "https://who.int/test", trust=0.8),
            EvidenceItem("test2", 0.7, "https://cdc.gov/test", trust=0.7),
            EvidenceItem("test3", 0.6, "https://nih.gov/test", trust=0.6),
        ]

        diversity = policy.calculate_diversity(evidence)
        assert diversity > 0.7  # High diversity with multiple domains

    def test_calculate_agreement_high(self, policy):
        """Test agreement calculation with high agreement."""
        evidence = [
            EvidenceItem("test", 0.8, "https://test.com", trust=0.8, stance="entails"),
            EvidenceItem("test2", 0.7, "https://test2.com", trust=0.7, stance="entails"),
            EvidenceItem("test3", 0.6, "https://test3.com", trust=0.6, stance="neutral"),
        ]

        agreement = policy.calculate_agreement(evidence)
        assert agreement == pytest.approx(1.0, rel=0.1)  # 3/3 non-contradicting

    def test_calculate_agreement_with_contradiction(self, policy):
        """Test agreement calculation with contradictions."""
        evidence = [
            EvidenceItem("test", 0.8, "https://test.com", trust=0.8, stance="entails"),
            EvidenceItem("test2", 0.7, "https://test2.com", trust=0.7, stance="contradicts"),
            EvidenceItem("test3", 0.6, "https://test3.com", trust=0.6, stance="neutral"),
        ]

        agreement = policy.calculate_agreement(evidence)
        assert agreement == pytest.approx(2 / 3, rel=0.1)  # 2/3 non-contradicting

    def test_apply_gating_rules_high_coverage(self, policy):
        """Test gating rules with high coverage."""
        assert policy.apply_gating_rules(coverage=0.7, diversity=0.5, agreement=0.8)

    def test_apply_gating_rules_medium_coverage_high_diversity(self, policy):
        """Test gating rules with medium coverage and high diversity."""
        assert policy.apply_gating_rules(coverage=0.5, diversity=0.8, agreement=0.8)

    def test_apply_gating_rules_insufficient(self, policy):
        """Test gating rules with insufficient evidence."""
        assert not policy.apply_gating_rules(coverage=0.3, diversity=0.4, agreement=0.8)

    def test_apply_gating_rules_requires_minimum_evidence_count(self, policy):
        """Low coverage should still fail on evidence-count gate."""
        assert not policy.apply_gating_rules(
            coverage=0.10,
            diversity=0.8,
            agreement=0.9,
            evidence_count=2,
            avg_relevance=0.30,
            strong_covered=0,
        )

    def test_adaptive_gate_passes_with_high_diversity(self, policy):
        assert policy.apply_gating_rules(
            coverage=0.25,
            diversity=1.0,
            agreement=1.0,
            evidence_count=2,
            avg_relevance=0.50,
            strong_covered=0,
        )

    def test_compute_adaptive_trust_single_claim(self, policy, sample_evidence):
        """Test adaptive trust computation for single-part claim."""
        claim = "COVID-19 vaccines are effective."

        result = policy.compute_adaptive_trust(claim, sample_evidence)

        assert result["num_subclaims"] == 1
        assert result["coverage"] > 0.5  # Should find matching evidence
        assert "is_sufficient" in result
        assert "verdict_state" in result
        assert "trust_post" in result

    def test_compute_adaptive_trust_multi_part_claim(self, policy, sample_evidence):
        """Test adaptive trust computation for multi-part claim."""
        claim = "COVID-19 vaccines are effective and they reduce hospitalization rates."

        result = policy.compute_adaptive_trust(claim, sample_evidence)

        assert result["num_subclaims"] == 2
        assert result["coverage"] == 1.0  # Both parts should be covered
        assert result["is_sufficient"]
        assert result["trust_post"] > 0

    def test_compute_adaptive_trust_insufficient_evidence(self, policy):
        """Test adaptive trust with insufficient evidence."""
        claim = "COVID-19 vaccines cause serious diseases and should be banned."
        poor_evidence = [EvidenceItem("Unrelated statement", 0.3, "https://test.com", trust=0.3, stance="neutral")]

        result = policy.compute_adaptive_trust(claim, poor_evidence)

        assert result["num_subclaims"] == 2
        assert result["coverage"] < 0.5  # Low coverage
        assert not result["is_sufficient"]
        assert result["verdict_state"] == "evidence_insufficiency"

    def test_compute_adaptive_trust_empty_evidence(self, policy):
        """Test adaptive trust with no evidence."""
        claim = "COVID-19 vaccines are effective."

        result = policy.compute_adaptive_trust(claim, [])

        assert result["num_subclaims"] == 1
        assert result["coverage"] == 0.0
        assert result["diversity"] == 0.0
        assert result["agreement"] == 0.0
        assert not result["is_sufficient"]
        assert result["trust_post"] == 0.0
