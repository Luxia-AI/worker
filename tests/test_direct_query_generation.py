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
    must_have = ["206", "106", "54", "52", "25", "100000", "tongue", "bones", "cells", "heart", "canada"]
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

    assert any('"' in q for q in queries), f"Missing quoted phrase constraint query: {queries}"
    assert any(
        ("filetype:pdf" in q) or ("intitle:" in q) or ("site:pubmed" in q) for q in queries
    ), f"Missing advanced operator query: {queries}"


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


def test_merge_subclaims_keeps_distinct_short_claims():
    ts = _init_trusted_search()
    parts = [
        "Vaccines do not cause autism",
        "Vaccines do not cause the flu",
    ]
    merged = ts.merge_subclaims(parts)
    assert len(merged) == 2
    assert merged[0] == parts[0]
    assert merged[1] == parts[1]


def test_merge_subclaims_expands_or_fragment_into_distinct_subclaim():
    ts = _init_trusted_search()
    parts = [
        "Vaccines do not cause autism",
        "or the flu",
    ]
    merged = ts.merge_subclaims(parts)
    assert len(merged) == 2
    assert merged[0] == "Vaccines do not cause autism"
    assert merged[1].lower() == "vaccines do not cause the flu"


@pytest.mark.asyncio
async def test_generate_queries_preserve_autism_and_flu_subclaims(monkeypatch):
    ts = _init_trusted_search()
    claim = "Vaccines do not cause autism or the flu."
    subclaims = ["Vaccines do not cause autism", "or the flu"]

    async def _no_llm(*args, **kwargs):
        return []

    monkeypatch.setattr(ts, "reformulate_queries", _no_llm)

    queries = await ts.generate_search_queries(
        post_text=claim,
        failed_entities=[],
        max_queries=6,
        subclaims=subclaims,
        entities=["vaccines", "autism", "flu"],
    )

    joined = " | ".join(queries).lower()
    assert "autism" in joined
    assert ("flu" in joined) or ("influenza" in joined)


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


def test_simplify_query_for_serper_fallback_removes_strict_operators():
    ts = _init_trusted_search()
    q = (
        'intitle:"meta-analysis" '
        '("diet rich fruits helps prevent noncommuni" OR "diet rich fruits") '
        '"noncommunicable diseases" statistics filetype:pdf'
    )
    simplified = ts._simplify_query_for_fallback(q)
    assert "intitle:" not in simplified
    assert "filetype:" not in simplified
    assert "(" not in simplified and ")" not in simplified


def test_domain_specific_queries_for_nutrition_claim():
    ts = _init_trusted_search()
    claim = (
        "A diet rich in fruits, vegetables, and low in saturated fats helps prevent "
        "noncommunicable diseases like diabetes and cancer."
    )
    qs = ts._build_domain_specific_queries(claim)
    joined = " | ".join(qs).lower()
    assert qs
    assert ("fruit" in joined) or ("vegetable" in joined)
    assert "saturated" in joined
    assert any(k in joined for k in ["systematic review", "meta-analysis", "clinical study", "guideline"])


def test_antibiotics_cold_flu_subclaim_queries_are_anchor_aligned():
    ts = _init_trusted_search()
    subclaim = "Antibiotics do not cure colds because they do not kill viruses that cause cold and flu."
    qs = ts._build_subclaim_anchor_queries(
        subclaim,
        entities=["antibiotics", "cold", "flu", "viruses"],
    )
    joined = " | ".join(qs).lower()
    assert "antibiotics" in joined
    assert any(k in joined for k in ["cold", "flu", "virus"])
    assert any(k in joined for k in ["systematic review", "meta-analysis", "no association", "pubmed"])


def test_sugar_hyperactivity_subclaim_queries_are_anchor_aligned():
    ts = _init_trusted_search()
    subclaim = "Studies show no link between sugar consumption and hyperactivity."
    qs = ts._build_subclaim_anchor_queries(
        subclaim,
        entities=["sugar", "hyperactivity", "children"],
    )
    joined = " | ".join(qs).lower()
    assert "sugar" in joined
    assert "hyperactivity" in joined
    assert any(k in joined for k in ["systematic review", "meta-analysis", "no association", "pubmed"])


def test_merge_subclaims_expands_do_not_cure_fragment():
    ts = _init_trusted_search()
    parts = [
        "Antibiotics do not cure colds",
        "or the flu",
    ]
    merged = ts.merge_subclaims(parts)
    assert len(merged) == 2
    assert merged[0] == "Antibiotics do not cure colds"
    assert merged[1].lower() == "antibiotics do not cure the flu"


def test_subclaim_anchors_do_not_bleed_unrelated_claim_entities():
    ts = _init_trusted_search()
    subclaim = "dim light may cause eye strain"
    # "permanent damage" belongs to a different subclaim and should not bleed in.
    anchors = ts._extract_subclaim_anchors(
        subclaim,
        entities=["eye strain", "permanent damage", "light"],
    )
    joined = " | ".join(anchors).lower()
    assert "eye strain" in joined
    assert "permanent damage" not in joined


def test_negation_tokens_not_emitted_as_raw_anchor_terms():
    ts = _init_trusted_search()
    subclaim = "detox diets are not scientifically supported"
    anchors = ts._extract_subclaim_anchors(subclaim, entities=["detox diets", "scientifically supported"])
    joined = " | ".join(anchors).lower()
    assert "not" not in anchors
    assert "no" not in anchors
    assert "detox diets" in joined
