from app.services.verdict.v2.posterior import compute_posteriors_v2
from app.services.verdict.v2.types import EvidenceScoreV2


def _make_score(
    support: float = 0.0,
    contradict: float = 0.0,
    neutral: float = 0.0,
    weight: float = 1.0,
    admissible: bool = True,
    domain: str = "example.org",
) -> EvidenceScoreV2:
    return EvidenceScoreV2(
        support_score=support,
        contradict_score=contradict,
        neutral_score=neutral,
        nli_entail_prob=support,
        nli_contradict_prob=contradict,
        nli_neutral_prob=neutral,
        admissible=admissible,
        weight=weight,
        source_domain=domain,
    )


class TestSufficiencyNotCrushedByEntropy:
    """After the fix, near-maximal retrieval entropy should not collapse sufficiency
    below the policy thresholds when directional mass is present."""

    def test_moderate_directional_mass_yields_reasonable_sufficiency(self):
        # Simulates v3.3-like inputs: coverage 0.35, diversity 1.0,
        # support+refute ~0.30, entropy ~0.98.
        scores = [
            _make_score(support=0.12, contradict=0.18, neutral=0.70, weight=0.50, domain="a.org"),
            _make_score(support=0.15, contradict=0.15, neutral=0.70, weight=0.45, domain="b.org"),
            _make_score(support=0.10, contradict=0.20, neutral=0.70, weight=0.48, domain="c.org"),
            _make_score(support=0.14, contradict=0.16, neutral=0.70, weight=0.47, domain="d.org"),
            _make_score(support=0.11, contradict=0.19, neutral=0.70, weight=0.46, domain="e.org"),
        ]
        post = compute_posteriors_v2(scores, coverage=0.35, diversity=1.0)
        assert post["sufficiency"] >= 0.52, (
            f"Sufficiency {post['sufficiency']:.4f} should be >= 0.52 "
            f"with moderate directional mass and high entropy"
        )
        assert post["p_unverifiable"] < 0.90, (
            f"p_unverifiable {post['p_unverifiable']:.4f} should not completely dominate "
            f"when directional evidence exists (was ~0.89 before fix)"
        )

    def test_strong_support_evidence_produces_high_sufficiency(self):
        scores = [
            _make_score(support=0.80, contradict=0.05, neutral=0.15, weight=0.90, domain="nih.gov"),
            _make_score(support=0.75, contradict=0.10, neutral=0.15, weight=0.85, domain="who.int"),
        ]
        post = compute_posteriors_v2(scores, coverage=0.80, diversity=0.75)
        assert post["sufficiency"] >= 0.65
        assert post["p_true"] > post["p_unverifiable"]

    def test_strong_refute_evidence_produces_false_leaning_posterior(self):
        scores = [
            _make_score(support=0.05, contradict=0.85, neutral=0.10, weight=0.90, domain="nih.gov"),
            _make_score(support=0.08, contradict=0.78, neutral=0.14, weight=0.88, domain="who.int"),
        ]
        post = compute_posteriors_v2(scores, coverage=0.90, diversity=0.80)
        assert post["p_false"] > post["p_true"]
        assert post["p_false"] > post["p_unverifiable"]


class TestSafetyPreserved:
    """Genuinely insufficient evidence should still yield low sufficiency
    and high UNVERIFIABLE posterior."""

    def test_very_low_coverage_and_diversity_yields_low_sufficiency(self):
        scores = [
            _make_score(support=0.10, contradict=0.05, neutral=0.85, weight=0.10, domain="x.org"),
        ]
        post = compute_posteriors_v2(scores, coverage=0.05, diversity=0.10)
        assert post["sufficiency"] < 0.40, (
            f"Sufficiency {post['sufficiency']:.4f} should remain low "
            f"when coverage and diversity are genuinely poor"
        )
        assert post["p_unverifiable"] > 0.60

    def test_no_evidence_yields_maximal_uncertainty(self):
        post = compute_posteriors_v2([], coverage=0.0, diversity=0.0)
        assert post["sufficiency"] < 0.30
        assert post["p_unverifiable"] > 0.80

    def test_all_neutral_evidence_favors_unverifiable(self):
        scores = [
            _make_score(support=0.05, contradict=0.05, neutral=0.90, weight=0.30, domain="a.org"),
            _make_score(support=0.03, contradict=0.02, neutral=0.95, weight=0.25, domain="b.org"),
        ]
        post = compute_posteriors_v2(scores, coverage=0.20, diversity=0.50)
        assert post["p_unverifiable"] > max(post["p_true"], post["p_false"])


class TestEntropyCapBehavior:
    """Directional-aware entropy capping should activate only when directional mass >= 0.25."""

    def test_entropy_capped_when_directional_mass_above_threshold(self):
        # High directional mass (support=0.50, contradict=0.20 -> mass=0.70)
        scores = [
            _make_score(support=0.50, contradict=0.20, neutral=0.30, weight=0.80, domain="a.org"),
            _make_score(support=0.45, contradict=0.25, neutral=0.30, weight=0.75, domain="b.org"),
        ]
        post_high = compute_posteriors_v2(scores, coverage=0.50, diversity=0.80)

        # Low directional mass (support=0.05, contradict=0.05 -> mass=0.10)
        scores_low = [
            _make_score(support=0.05, contradict=0.05, neutral=0.90, weight=0.80, domain="a.org"),
            _make_score(support=0.05, contradict=0.05, neutral=0.90, weight=0.75, domain="b.org"),
        ]
        post_low = compute_posteriors_v2(scores_low, coverage=0.50, diversity=0.80)

        # High directional mass should have higher sufficiency due to entropy capping
        assert post_high["sufficiency"] > post_low["sufficiency"]
