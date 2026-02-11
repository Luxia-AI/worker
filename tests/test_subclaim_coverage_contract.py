from app.services.ranking.subclaim_coverage import compute_subclaim_coverage
from app.services.ranking.trust_ranker import EvidenceItem


def test_autism_only_evidence_cannot_fully_cover_autism_and_flu_claim():
    subclaims = [
        "Vaccines do not cause autism",
        "Vaccines do not cause the flu",
    ]
    evidence = [
        EvidenceItem(
            statement="Vaccines are not associated with autism.",
            semantic_score=0.95,
            source_url="https://pubmed.ncbi.nlm.nih.gov/24814559/",
            stance="entails",
            trust=0.9,
        )
    ]

    coverage = compute_subclaim_coverage(subclaims, evidence, partial_weight=0.5)

    assert coverage["coverage"] <= 0.5
    details = coverage["details"]
    assert details[0]["status"] in {"STRONGLY_VALID", "PARTIALLY_VALID"}
    assert details[1]["status"] != "STRONGLY_VALID"


def test_contradicting_stance_does_not_count_as_subclaim_coverage():
    subclaims = ["Vaccines do not cause autism"]
    evidence = [
        EvidenceItem(
            statement="Some studies suggest vaccines may cause autism.",
            semantic_score=0.92,
            source_url="https://example.org/claim",
            stance="contradicts",
            trust=0.9,
        )
    ]

    coverage = compute_subclaim_coverage(subclaims, evidence, partial_weight=0.5)
    assert coverage["coverage"] == 0.0
    assert coverage["details"][0]["status"] == "INVALID"
    assert coverage["details"][0]["contradicted"] is True


def test_negation_polarity_examples_for_segment_coverage():
    # "Antibiotics do not work against viruses" -> TRUE (supported by negative evidence form)
    c1 = compute_subclaim_coverage(
        ["Antibiotics do not work against viruses"],
        [
            EvidenceItem(
                statement="Antibiotics do not treat viral infections.",
                semantic_score=0.95,
                source_url="https://www.cdc.gov/example",
                stance="neutral",
                trust=0.9,
            )
        ],
    )
    assert c1["details"][0]["status"] in {"STRONGLY_VALID", "PARTIALLY_VALID"}

    # "Antibiotics work against viruses" -> FALSE (contradicted by negative evidence form)
    c2 = compute_subclaim_coverage(
        ["Antibiotics work against viruses"],
        [
            EvidenceItem(
                statement="Antibiotics do not treat viral infections.",
                semantic_score=0.95,
                source_url="https://www.cdc.gov/example",
                stance="neutral",
                trust=0.9,
            )
        ],
    )
    assert c2["details"][0]["status"] == "INVALID"
    assert c2["details"][0]["contradicted"] is True

    # "Handwashing with soap reduces spread" -> TRUE
    c3 = compute_subclaim_coverage(
        ["Handwashing with soap reduces spread of infectious diseases"],
        [
            EvidenceItem(
                statement="Handwashing with soap reduces the spread of infectious diseases.",
                semantic_score=0.95,
                source_url="https://www.who.int/example",
                stance="neutral",
                trust=0.9,
            )
        ],
    )
    assert c3["details"][0]["status"] in {"STRONGLY_VALID", "PARTIALLY_VALID"}

    # "Handwashing does not reduce spread" -> FALSE
    c4 = compute_subclaim_coverage(
        ["Handwashing does not reduce spread of infectious diseases"],
        [
            EvidenceItem(
                statement="Handwashing with soap reduces the spread of infectious diseases.",
                semantic_score=0.95,
                source_url="https://www.who.int/example",
                stance="neutral",
                trust=0.9,
            )
        ],
    )
    assert c4["details"][0]["status"] == "INVALID"
    assert c4["details"][0]["contradicted"] is True
