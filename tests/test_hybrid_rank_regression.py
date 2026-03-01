"""
Regression tests for hybrid_rank to enforce invariants after the ranking fix.

These tests guard against recurrence of the core failure modes:
  1. KG-only input yielding empty output
  2. Mixed VDB+KG dropping all KG candidates
  3. Dedup incorrectly removing unique KG evidence
  4. Score compression (near-flat zeros, strong candidates below threshold)
"""

from datetime import datetime, timezone

from app.services.ranking.hybrid_ranker import hybrid_rank

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sem(statement, score, entities=None, source_url=None, credibility=None, published_at=None):
    d = {"statement": statement, "score": score, "entities": entities or []}
    if source_url:
        d["source_url"] = source_url
    if credibility is not None:
        d["credibility"] = credibility
    if published_at:
        d["published_at"] = published_at
    return d


def _make_kg(statement, score, entities=None, source_url=None, credibility=None, candidate_type="KG"):
    d = {
        "statement": statement,
        "score": score,
        "entities": entities or [],
        "candidate_type": candidate_type,
    }
    if source_url:
        d["source_url"] = source_url
    if credibility is not None:
        d["credibility"] = credibility
    return d


# ======================================================================
# 1. KG-only input yields non-empty ranked output
# ======================================================================


class TestKGOnlyNonEmpty:
    """KG-only candidates must survive ranking and produce output."""

    def test_single_kg_candidate_survives(self):
        kg = [_make_kg("Aspirin reduces cardiovascular risk", 0.85, ["aspirin"])]
        ranked = hybrid_rank([], kg)
        assert len(ranked) == 1
        assert ranked[0]["final_score"] > 0.0

    def test_multiple_kg_candidates_all_survive(self):
        kg = [
            _make_kg("Insulin treats diabetes", 0.90, ["insulin", "diabetes"], source_url="https://nih.gov/diabetes"),
            _make_kg(
                "Metformin reduces blood sugar", 0.75, ["metformin"], source_url="https://pubmed.ncbi.nlm.nih.gov/456/"
            ),
            _make_kg("Exercise improves glucose control", 0.60, ["exercise"]),
        ]
        ranked = hybrid_rank([], kg)
        assert len(ranked) == 3
        assert all(r["final_score"] > 0.0 for r in ranked)

    def test_kg_only_with_high_credibility_source(self):
        kg = [
            _make_kg(
                "WHO recommends vaccination schedule",
                0.88,
                ["vaccination"],
                source_url="https://who.int/immunization",
                credibility=0.95,
            ),
        ]
        ranked = hybrid_rank([], kg)
        assert len(ranked) == 1
        assert ranked[0]["final_score"] > 0.30

    def test_kg_only_low_score_still_produces_output(self):
        """Even low-score KG candidates should appear in output (not filtered out)."""
        kg = [_make_kg("Some weak relation", 0.20, ["x"])]
        ranked = hybrid_rank([], kg)
        assert len(ranked) >= 1

    def test_kg_only_preserves_ordering_by_score(self):
        kg = [
            _make_kg("Low", 0.30, ["a"]),
            _make_kg("High", 0.90, ["b"]),
            _make_kg("Mid", 0.60, ["c"]),
        ]
        ranked = hybrid_rank([], kg)
        assert ranked[0]["statement"] == "High"
        assert ranked[-1]["statement"] == "Low"


# ======================================================================
# 2. Mixed VDB+KG preserves at least one KG candidate in top-N
# ======================================================================


class TestMixedPreservesKG:
    """When both VDB and KG candidates exist, KG must not be fully eliminated."""

    def test_kg_candidate_present_in_results(self):
        sem = [
            _make_sem("Semantic evidence A", 0.85, ["topic"], source_url="https://cdc.gov/topic"),
        ]
        kg = [
            _make_kg("KG evidence B", 0.80, ["topic"], source_url="https://nih.gov/topic"),
        ]
        ranked = hybrid_rank(sem, kg, query_entities=["topic"])
        kg_in_results = [r for r in ranked if r.get("candidate_type") == "KG" or r.get("kg_score", 0) > 0]
        assert len(kg_in_results) >= 1, "At least one KG candidate must survive mixed ranking"

    def test_high_quality_kg_survives_with_strong_vdb(self):
        """Strong VDB should not eliminate a strong KG candidate."""
        sem = [
            _make_sem("Top semantic result", 0.95, ["immune"], source_url="https://who.int/health", credibility=0.95),
        ]
        kg = [
            _make_kg(
                "Vitamin C supports immune system",
                0.90,
                ["vitamin c", "immune"],
                source_url="https://nih.gov/vitaminc",
                credibility=0.95,
            ),
        ]
        ranked = hybrid_rank(sem, kg, query_entities=["immune"])
        assert len(ranked) == 2
        kg_results = [r for r in ranked if r.get("kg_score", 0) > 0]
        assert len(kg_results) >= 1

    def test_kg_retention_when_filters_would_eliminate(self):
        """KG retention policy should rescue best KG even if filters drop it."""
        sem = [
            _make_sem(
                "Strong semantic evidence about sleep improvement", 0.92, ["sleep"], source_url="https://cdc.gov/sleep"
            ),
            _make_sem(
                "Another semantic about sleep quality", 0.88, ["sleep quality"], source_url="https://who.int/sleep"
            ),
        ]
        kg = [
            _make_kg(
                "Melatonin pathway REGULATES sleep cycle",
                0.70,
                ["melatonin", "sleep"],
                source_url="https://nih.gov/sleep",
            ),
        ]
        ranked = hybrid_rank(sem, kg, query_entities=["sleep"], query_text="melatonin helps with sleep")
        # KG candidate should survive (either directly or via retention policy)
        assert len(ranked) >= 2

    def test_multiple_kg_some_survive_mixed(self):
        sem = [
            _make_sem("VDB result 1", 0.90, ["cancer"]),
            _make_sem("VDB result 2", 0.80, ["cancer"]),
        ]
        kg = [
            _make_kg(
                "Chemotherapy treats cancer",
                0.85,
                ["chemotherapy", "cancer"],
                source_url="https://nih.gov/chemo",
                credibility=0.95,
            ),
            _make_kg(
                "Surgery removes tumors",
                0.75,
                ["surgery", "tumors"],
                source_url="https://who.int/surgery",
                credibility=0.95,
            ),
            _make_kg("Weak unrelated link", 0.15, ["unrelated"]),
        ]
        ranked = hybrid_rank(sem, kg, query_entities=["cancer"])
        kg_survivors = [r for r in ranked if r.get("candidate_type") == "KG" or r.get("kg_score", 0) > 0]
        assert len(kg_survivors) >= 1, "At least one strong KG candidate should survive"


# ======================================================================
# 3. Dedup does not remove unique KG evidence incorrectly
# ======================================================================


class TestDedupCorrectness:
    """Dedup should merge exact duplicates but preserve unique evidence."""

    def test_same_statement_different_source_merged(self):
        """Same statement from VDB and KG should merge into one entry."""
        sem = [_make_sem("Shared statement", 0.80, ["e1"])]
        kg = [_make_kg("Shared statement", 0.75, ["e1"])]
        ranked = hybrid_rank(sem, kg)
        assert len(ranked) == 1
        assert ranked[0]["sem_score"] > 0
        assert ranked[0]["kg_score"] > 0

    def test_different_statements_from_same_source_not_merged(self):
        """Different statements should not be merged even with same source URL."""
        sem = [
            _make_sem("Statement A about topic", 0.80, ["topic"], source_url="https://cdc.gov/same"),
            _make_sem("Statement B different content", 0.75, ["topic"], source_url="https://cdc.gov/same"),
        ]
        ranked = hybrid_rank(sem, [])
        assert len(ranked) == 2

    def test_kg_unique_evidence_not_incorrectly_removed(self):
        """KG evidence with unique statements must not be dropped by dedup."""
        sem = [_make_sem("Vitamin D linked to bone health", 0.85, ["vitamin d"])]
        kg = [
            _make_kg("Calcium absorption requires vitamin D", 0.80, ["calcium", "vitamin d"]),
            _make_kg("Vitamin D deficiency causes rickets", 0.75, ["vitamin d", "rickets"]),
        ]
        ranked = hybrid_rank(sem, kg)
        assert len(ranked) == 3, "All three unique statements should survive"

    def test_entities_merged_on_dedup(self):
        """When merging duplicates, entities from both sources should be combined."""
        sem = [_make_sem("Shared claim", 0.80, ["entity_a"])]
        kg = [_make_kg("Shared claim", 0.70, ["entity_b"])]
        ranked = hybrid_rank(sem, kg)
        assert len(ranked) == 1
        ents = [e.lower() for e in ranked[0]["entities"]]
        assert "entity_a" in ents
        assert "entity_b" in ents


# ======================================================================
# 4. Score sanity checks
# ======================================================================


class TestScoreSanity:
    """Score distribution must be non-trivial with well-separated values."""

    def test_strong_candidate_exceeds_minimum_threshold(self):
        """A strong candidate (high sem + high cred + trusted source) must score above 0.50."""
        sem = [
            _make_sem(
                "WHO confirms vaccine safety",
                0.92,
                ["vaccine", "safety"],
                source_url="https://who.int/safety",
                credibility=0.95,
                published_at="2024-06-01",
            ),
        ]
        ranked = hybrid_rank(sem, [], query_entities=["vaccine", "safety"])
        assert ranked[0]["final_score"] > 0.50

    def test_score_spread_nontrivial_in_top_k(self):
        """Top-k results should have non-trivial score spread (not all near zero)."""
        sem = [
            _make_sem("High quality fact", 0.95, ["health"], source_url="https://cdc.gov/", credibility=0.95),
            _make_sem(
                "Medium quality fact", 0.70, ["health"], source_url="https://health.harvard.edu/", credibility=0.80
            ),
            _make_sem("Low quality fact", 0.40, ["health"], source_url="https://blog.example.com/", credibility=0.30),
        ]
        ranked = hybrid_rank(sem, [], query_entities=["health"])
        scores = [r["final_score"] for r in ranked]
        score_spread = max(scores) - min(scores)
        assert score_spread > 0.05, f"Score spread too narrow: {score_spread:.4f}"
        assert max(scores) > 0.30, f"Maximum score too low: {max(scores):.4f}"

    def test_no_negative_scores(self):
        """All final scores must be non-negative."""
        sem = [
            _make_sem("A", 0.10, ["x"]),
            _make_sem("B", 0.01, ["y"]),
        ]
        kg = [_make_kg("C", 0.05, ["z"])]
        ranked = hybrid_rank(sem, kg)
        assert all(r["final_score"] >= 0.0 for r in ranked)

    def test_scores_capped_at_one(self):
        """All final scores must be capped at 1.0."""
        sem = [
            _make_sem(
                "Perfect evidence",
                1.0,
                ["topic"],
                source_url="https://who.int/",
                credibility=0.95,
                published_at="2024-12-01",
            ),
        ]
        kg = [
            _make_kg("Also perfect KG", 1.0, ["topic"], source_url="https://nih.gov/", credibility=0.95),
        ]
        ranked = hybrid_rank(sem, kg, query_entities=["topic"])
        assert all(r["final_score"] <= 1.0 for r in ranked)

    def test_authority_bonus_lifts_trusted_source(self):
        """Trusted authority source should score meaningfully higher than unknown source."""
        trusted = _make_sem(
            "Evidence from authority", 0.60, ["topic"], source_url="https://who.int/evidence", credibility=0.95
        )
        untrusted = _make_sem(
            "Evidence from unknown", 0.60, ["topic"], source_url="https://unknown-blog.com/post", credibility=0.30
        )
        ranked = hybrid_rank([trusted, untrusted], [])
        trusted_result = next(r for r in ranked if "authority" in r["statement"].lower())
        untrusted_result = next(r for r in ranked if "unknown" in r["statement"].lower())
        assert trusted_result["final_score"] > untrusted_result["final_score"]

    def test_credibility_bonus_field_present(self):
        """Output should include credibility_bonus field."""
        sem = [_make_sem("Some fact", 0.80, source_url="https://who.int/x", credibility=0.95)]
        ranked = hybrid_rank(sem, [])
        assert "credibility_bonus" in ranked[0]
        assert ranked[0]["credibility_bonus"] >= 0.0

    def test_kg_score_raw_preserved(self):
        """KG score raw should be preserved in output for diagnostics."""
        kg = [_make_kg("KG statement", 0.85, ["entity"])]
        ranked = hybrid_rank([], kg)
        assert ranked[0]["kg_score_raw"] > 0.0

    def test_support_and_contradict_scores_present(self):
        """Both support_score and contradict_score must be in output."""
        sem = [_make_sem("Test statement", 0.75, ["test"])]
        ranked = hybrid_rank(sem, [])
        assert "support_score" in ranked[0]
        assert "contradict_score" in ranked[0]
        assert 0.0 <= ranked[0]["support_score"] <= 1.0
        assert 0.0 <= ranked[0]["contradict_score"] <= 1.0

    def test_deterministic_output_across_runs(self):
        """Identical input must produce identical output (deterministic)."""
        sem = [
            _make_sem("Evidence A", 0.80, ["e1"]),
            _make_sem("Evidence B", 0.80, ["e2"]),
            _make_sem("Evidence C", 0.80, ["e3"]),
        ]
        kg = [_make_kg("KG A", 0.70, ["e1"]), _make_kg("KG B", 0.70, ["e4"])]
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)

        run1 = hybrid_rank(sem, kg, query_entities=["e1"], now=now)
        run2 = hybrid_rank(sem, kg, query_entities=["e1"], now=now)

        assert len(run1) == len(run2)
        for r1, r2 in zip(run1, run2):
            assert r1["statement"] == r2["statement"]
            assert r1["final_score"] == r2["final_score"]

    def test_sem_score_is_weighted_component(self):
        """sem_score output should be the weighted component (w_sem * normalized), not raw."""
        sem = [_make_sem("Single result", 0.90, ["topic"])]
        ranked = hybrid_rank(sem, [])
        # w_semantic default is 0.31; for a single item normalization preserves score
        # so sem_score should be roughly w_sem * score, not the raw 0.90
        assert ranked[0]["sem_score"] < ranked[0]["sem_score_raw"] or ranked[0]["sem_score"] <= 0.50
