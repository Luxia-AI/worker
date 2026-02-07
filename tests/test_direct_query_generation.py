import pytest

from app.services.corrective.trusted_search import TrustedSearch
from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy
from app.services.ranking.trust_ranker import EvidenceItem
from app.services.verdict.verdict_generator import VerdictGenerator


def _init_trusted_search() -> TrustedSearch:
    # Avoid __init__ which requires external API credentials/LLM
    return TrustedSearch.__new__(TrustedSearch)


@pytest.mark.asyncio
async def test_direct_queries_cover_key_terms_and_numbers():
    ts = _init_trusted_search()
    claim = (
        "Just like fingerprints, every individual has a unique tongue print. "
        "Adults have 206 bones, with 106 of them located in the hands and feet "
        "(54 in the hands, 52 in the feet). "
        "Your body produces 25 million new cells every second, which is roughly the population of Canada. "
        "The human heart beats around 100,000 times per day."
    )

    subclaims = AdaptiveTrustPolicy().decompose_claim(claim)
    merged = ts.merge_subclaims(subclaims)
    queries = await ts.generate_search_queries(
        post_text=claim,
        failed_entities=[],
        max_queries=6,
        subclaims=merged,
        entities=[],
    )

    assert len(queries) == 6
    must_have = ["206", "106", "54", "52", "25", "100000", "tongue", "bones", "cells", "heart"]
    for q in queries:
        assert any(term in q for term in must_have), f"Query not direct to claim: {q}"


@pytest.mark.asyncio
async def test_advanced_queries_include_boolean_and_operator_constraints():
    ts = _init_trusted_search()
    claim = "The human heart beats around 100,000 times per day in adults."

    subclaims = AdaptiveTrustPolicy().decompose_claim(claim)
    merged = ts.merge_subclaims(subclaims)
    queries = await ts.generate_search_queries(
        post_text=claim,
        failed_entities=[],
        max_queries=6,
        subclaims=merged,
        entities=["human heart", "adults"],
    )

    assert any(" or " in q.lower() for q in queries), f"Missing boolean synonym query: {queries}"
    assert any("filetype:pdf" in q for q in queries), f"Missing filetype query: {queries}"
    assert any("intitle:" in q for q in queries), f"Missing intitle query: {queries}"


def test_research_instruction_query_adds_exclusion_filters_for_adult_population():
    ts = _init_trusted_search()
    q = ts._build_research_instruction_query(
        "Adult human heart rate statistics",
        entities=["heart rate", "adult human"],
    )
    assert "-fetal" in q
    assert "-pediatric" in q


def test_merge_subclaims_conjunction():
    ts = _init_trusted_search()
    parts = [
        "Adults have 206 bones, with 106 of them located in the hands",
        "and feet (54 in the hands, 52 in the feet).",
    ]
    merged = ts.merge_subclaims(parts)
    assert len(merged) == 1
    assert "hands and feet" in merged[0].lower()


def test_truthfulness_segment_based_high_support():
    vg = VerdictGenerator.__new__(VerdictGenerator)

    claim = "Adults have 206 bones. The human heart beats around 100,000 times per day."
    evidence = [
        {
            "statement": "The adult human body has 206 bones.",
            "final_score": 0.90,
            "credibility": 0.95,
            "source_url": "https://www.nih.gov/",
        },
        {
            "statement": "The human heart beats about 100,000 times per day.",
            "final_score": 0.90,
            "credibility": 0.95,
            "source_url": "https://www.cdc.gov/",
        },
    ]

    truthfulness = vg._calculate_truthfulness_from_evidence(claim, evidence)
    assert truthfulness >= 80.0


def test_adaptive_coverage_relaxed_threshold():
    policy = AdaptiveTrustPolicy()
    subclaims = ["Adults have 206 bones in the human body."]
    evidence = [
        EvidenceItem(
            statement="The adult human body has 206 bones.",
            semantic_score=0.65,
            source_url="https://www.nih.gov/",
            trust=0.65,
        )
    ]

    coverage = policy.calculate_coverage(subclaims, evidence)
    assert coverage > 0.0
