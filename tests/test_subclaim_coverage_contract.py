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
    assert details[1]["status"] == "UNKNOWN"


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
    assert coverage["details"][0]["status"] == "UNKNOWN"
    assert coverage["details"][0]["contradicted"] is True
