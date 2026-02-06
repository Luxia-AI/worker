from app.services.retrieval.evidence_gate import filter_candidates_for_count_claim


def test_count_claim_gate_filters_non_numeric():
    sem = [
        {"statement": "Bone marrow is inside bones.", "fact_type": "definition"},
        {"statement": "The adult human skeleton has 206 bones.", "fact_type": "count"},
    ]
    kg = [
        {"statement": "skeleton has 206 bones"},
        {"statement": "vaccines prevent disease"},
    ]

    filtered_sem, filtered_kg = filter_candidates_for_count_claim(
        sem,
        kg,
        "Adults have 206 bones.",
        ["bones", "skeleton"],
    )

    assert len(filtered_sem) == 1  # nosec
    assert "206 bones" in filtered_sem[0]["statement"]  # nosec
    assert len(filtered_kg) == 1  # nosec
