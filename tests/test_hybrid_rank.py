from datetime import datetime, timezone

from app.services.ranking.hybrid_ranker import hybrid_rank


def test_empty_inputs():
    ranked = hybrid_rank([], [])
    assert ranked == []  # nosec


def test_semantic_only_ranking():
    sem = [
        {"statement": "A", "score": 0.2, "entities": []},
        {"statement": "B", "score": 0.9, "entities": []},
        {"statement": "C", "score": 0.5, "entities": []},
    ]
    ranked = hybrid_rank(sem, [])
    assert ranked[0]["statement"] == "B"  # nosec  # highest semantic
    assert ranked[-1]["statement"] == "A"  # nosec


def test_kg_only_ranking():
    kg = [
        {"statement": "X", "score": 0.1, "entities": []},
        {"statement": "Y", "score": 0.9, "entities": []},
    ]
    ranked = hybrid_rank([], kg)
    assert ranked[0]["statement"] == "Y"  # nosec
    assert ranked[-1]["statement"] == "X"  # nosec


def test_merge_semantic_and_kg_duplicate():
    sem = [{"statement": "Shared", "score": 0.7, "entities": ["vitamin d"]}]
    kg = [{"statement": "Shared", "score": 0.9, "entities": ["vitamin d"]}]

    ranked = hybrid_rank(sem, kg)
    assert len(ranked) == 1  # nosec  # merged
    top = ranked[0]
    assert top["sem_score"] > 0  # nosec
    assert top["kg_score"] > 0  # nosec


def test_entity_overlap_scoring():
    sem = [{"statement": "Vitamin D improves immunity.", "score": 0.8, "entities": ["vitamin d", "immunity"]}]

    ranked = hybrid_rank(semantic_results=sem, kg_results=[], query_entities=["vitamin d"])

    assert ranked[0]["entity_overlap"] > 0.0  # nosec
    assert ranked[0]["final_score"] > ranked[0]["sem_score"]  # nosec  # overlap boost


def test_recency_scoring():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)

    recent = {"statement": "Recent evidence", "score": 0.5, "entities": [], "published_at": "2024-12-30T00:00:00+00:00"}
    old = {"statement": "Old evidence", "score": 0.5, "entities": [], "published_at": "2010-01-01T00:00:00+00:00"}

    ranked = hybrid_rank(semantic_results=[recent, old], kg_results=[], now=now)

    assert ranked[0]["statement"] == "Recent evidence"  # nosec
    assert ranked[0]["recency"] > ranked[1]["recency"]  # nosec


def test_credibility_scoring_authoritative_sources():
    sem = [
        {"statement": "Trusted evidence", "score": 0.1, "entities": [], "source_url": "https://who.int/study"},
        {"statement": "Less trusted", "score": 0.9, "entities": [], "source_url": "https://randomblog.com/post"},
    ]

    ranked = hybrid_rank(sem, [])
    assert ranked[0]["statement"] == "Trusted evidence"  # nosec  # WHO credibility outranks raw similarity


def test_weights_shift_results():
    # B has better semantic score
    sem = [
        {"statement": "A", "score": 0.1, "entities": ["x"]},
        {"statement": "B", "score": 0.9, "entities": ["y"]},
    ]
    # A has strong KG score
    kg = [{"statement": "A", "score": 1.0, "entities": ["x"]}]

    # Default weights → B first (semantic strong)
    ranked_default = hybrid_rank(sem, kg)
    assert ranked_default[0]["statement"] == "B"  # nosec

    # Increase KG weight heavily
    ranked_kg_heavy = hybrid_rank(sem, kg, weights={"w_kg": 0.8, "w_semantic": 0.1})
    assert ranked_kg_heavy[0]["statement"] == "A"  # nosec


def test_sort_stability_on_equal_scores():
    sem = [
        {"statement": "Alpha", "score": 0.5, "entities": []},
        {"statement": "Beta", "score": 0.5, "entities": []},
        {"statement": "Gamma", "score": 0.5, "entities": []},
    ]

    ranked = hybrid_rank(sem, [])

    # When equal, fallback alphabetical
    assert [r["statement"] for r in ranked] == ["Alpha", "Beta", "Gamma"]  # nosec


def test_missing_fields_handled_gracefully():
    sem = [
        {"statement": "No entities but valid", "score": 0.3},
        {"statement": "No score", "entities": ["a"]},  # score missing
    ]
    ranked = hybrid_rank(sem, [])
    assert len(ranked) == 2  # nosec
    assert all("final_score" in r for r in ranked)  # nosec
