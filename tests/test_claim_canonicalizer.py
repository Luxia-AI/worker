import asyncio

from app.services.common.claim_canonicalizer import CanonicalClaim, CanonicalClaimSegment, ClaimCanonicalizer
from app.services.verdict.verdict_generator import VerdictGenerator


def test_canonicalizer_accepts_faithful_rule_parse(monkeypatch):
    monkeypatch.setenv("REFUTE_NLI_ENABLED", "false")
    monkeypatch.setenv("CLAIM_CANONICALIZATION_ENABLED", "true")
    monkeypatch.setenv("CLAIM_CANONICAL_LLM_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("CLAIM_CANONICAL_DRIFT_GUARD_ENABLED", "true")
    canonicalizer = ClaimCanonicalizer()

    result = asyncio.run(canonicalizer.canonicalize_claim("Antibiotics treat bacterial infections."))
    assert result.segments
    assert all(seg.original_text for seg in result.segments)
    assert any(seg.canonical_accepted for seg in result.segments)
    assert 0.0 <= result.canonical_accept_rate <= 1.0


def test_canonicalizer_drift_guard_rejects_semantic_shift(monkeypatch):
    monkeypatch.setenv("REFUTE_NLI_ENABLED", "false")
    monkeypatch.setenv("CLAIM_CANONICALIZATION_ENABLED", "true")
    monkeypatch.setenv("CLAIM_CANONICAL_LLM_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("CLAIM_CANONICAL_DRIFT_GUARD_ENABLED", "true")
    canonicalizer = ClaimCanonicalizer()

    segment = CanonicalClaimSegment(
        segment_id="s1",
        original_text="Vitamin C reduces symptom duration.",
        normalized_text="Vitamin C increases symptom duration.",
        subject="Vitamin C",
        predicate="increases",
        object="symptom duration",
        polarity="positive",
        quantifier="",
        comparator="",
        numeric_value="",
        unit="",
        population="",
        timeframe="",
        modality="",
        parse_confidence=0.9,
        canonical_source="llm",
    )
    guarded = canonicalizer._drift_guard(segment)
    assert guarded.canonical_accepted is False
    assert "drift_guard_failed" in guarded.canonical_rejected_reason
    assert guarded.normalized_text.rstrip(".") == guarded.original_text.rstrip(".")


def test_policy_v3_projection_is_monotonic():
    vg = VerdictGenerator.__new__(VerdictGenerator)
    base_payload = {
        "evidence_map": [
            {"relevance_score": 0.8, "credibility": 0.9},
            {"relevance_score": 0.7, "credibility": 0.8},
        ],
        "policy_trace": [],
        "claim_canonical_segments": [],
        "canonical_rejections": [],
        "alignment_orig": 0.7,
        "alignment_canon": 0.6,
        "alignment_fused": 0.67,
        "trust_threshold_met": True,
        "analysis_counts": {},
    }
    out_support = vg._apply_policy_v3_deterministic_projection(
        payload=base_payload,
        support_mass=1.4,
        contradict_mass=0.2,
        neutral_mass=0.2,
        sufficiency_score=0.8,
    )
    out_contradict = vg._apply_policy_v3_deterministic_projection(
        payload=base_payload,
        support_mass=0.2,
        contradict_mass=1.4,
        neutral_mass=0.2,
        sufficiency_score=0.8,
    )
    assert out_support["truth_score_binary"] > out_contradict["truth_score_binary"]
    assert out_support["verdict_binary"] == "TRUE"
    assert out_contradict["verdict_binary"] == "FALSE"
    assert out_support["confidence"] >= 0.05
    assert out_contradict["confidence"] >= 0.05


def test_dual_track_query_templates_have_bounded_budget():
    canonical_claim = CanonicalClaim(
        claim_original="Antibiotics treat bacterial infections.",
        segments=[
            CanonicalClaimSegment(
                segment_id="s1",
                original_text="Antibiotics treat bacterial infections",
                normalized_text="antibiotics treat bacterial infections",
                subject="antibiotics",
                predicate="treat",
                object="bacterial infections",
                polarity="positive",
                quantifier="",
                comparator="",
                numeric_value="",
                unit="",
                population="",
                timeframe="",
                modality="",
                parse_confidence=0.9,
                canonical_source="rules",
                canonical_accepted=True,
            )
        ],
    )
    built = ClaimCanonicalizer.build_dual_track_queries(canonical_claim, max_per_segment=8)
    merged = built["queries_merged"]
    assert merged
    assert len(merged) <= 8
    assert len(built["queries_original"]) <= 5
    assert len(built["queries_canonical"]) <= 3


def test_recommendation_authority_claim_role_assignment(monkeypatch):
    monkeypatch.setenv("REFUTE_NLI_ENABLED", "false")
    monkeypatch.setenv("CLAIM_CANONICALIZATION_ENABLED", "true")
    monkeypatch.setenv("CLAIM_CANONICAL_LLM_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("CLAIM_CANONICAL_DRIFT_GUARD_ENABLED", "false")
    canonicalizer = ClaimCanonicalizer()

    claim = "Eating junk food daily is the primary recommendation of the WHO."
    result = asyncio.run(canonicalizer.canonicalize_claim(claim))
    assert result.segments
    seg = result.segments[0]
    assert "who" in seg.subject.lower()
    assert seg.predicate in {"recommends", "advises"}
    assert seg.predicate_family == "recommendation"
    assert "eating junk food daily" in seg.object.lower()


def test_recommendation_authority_queries_are_sane_and_grammar_safe():
    canonical_claim = CanonicalClaim(
        claim_original="Eating junk food daily is the primary recommendation of the WHO.",
        segments=[
            CanonicalClaimSegment(
                segment_id="s1",
                original_text="Eating junk food daily is the primary recommendation of the WHO",
                normalized_text="WHO recommends eating junk food daily",
                subject="WHO",
                predicate="recommends",
                object="eating junk food daily",
                polarity="positive",
                quantifier="",
                comparator="",
                numeric_value="",
                unit="",
                population="",
                timeframe="",
                modality="",
                parse_confidence=0.95,
                canonical_source="rules",
                predicate_family="recommendation",
                canonical_accepted=True,
            )
        ],
    )
    built = ClaimCanonicalizer.build_dual_track_queries(canonical_claim, max_per_segment=8)
    merged = [q.lower() for q in built["queries_merged"]]
    assert any("who recommends eating junk food daily" in q for q in merged)
    assert any("who advises against eating junk food daily" in q for q in merged)
    assert all("does not is" not in q for q in merged)
