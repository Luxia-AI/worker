from app.services.ranking.subclaim_coverage import compute_subclaim_coverage
from app.services.ranking.trust_ranker import EvidenceItem
from app.shared.anchor_extraction import AnchorExtractor


def test_anchor_extraction_filters_stopwords_and_junk_terms(monkeypatch):
    extractor = AnchorExtractor(llm_service=None)
    monkeypatch.setattr(extractor, "_extract_with_llm_sync", lambda claim, subclaims: {})

    claim = "Antibiotics do not work against viruses."
    result = extractor.extract_for_claim(
        claim=claim,
        subclaims=[claim],
        entity_hints=["antibiotics", "viruses"],
    )
    anchors = result.anchors_by_subclaim[claim]
    assert anchors
    assert all("against" not in anchor for anchor in anchors)
    assert all(anchor != "work" for anchor in anchors)


def test_anchor_extraction_contains_antibiotics_and_viral_terms(monkeypatch):
    extractor = AnchorExtractor(llm_service=None)
    monkeypatch.setattr(extractor, "_extract_with_llm_sync", lambda claim, subclaims: {})

    claim = "Antibiotics do not work against viruses."
    result = extractor.extract_for_claim(
        claim=claim,
        subclaims=[claim],
        entity_hints=["antibiotics", "viral infection"],
    )
    anchors = result.anchors_by_subclaim[claim]
    assert any("antibiotic" in anchor for anchor in anchors)
    assert any(("virus" in anchor) or ("viral" in anchor) for anchor in anchors)


def test_coverage_marks_strong_when_evidence_has_antibiotic_viral_negation(monkeypatch):
    extractor = AnchorExtractor(llm_service=None)
    monkeypatch.setattr(extractor, "_extract_with_llm_sync", lambda claim, subclaims: {})
    claim = "Antibiotics do not work against viruses."
    anchors = extractor.extract_for_claim(claim=claim, subclaims=[claim], entity_hints=["antibiotics", "viruses"])

    evidence = [
        EvidenceItem(
            statement="Antibiotics do not treat viral infections.",
            semantic_score=0.92,
            source_url="https://www.cdc.gov/antibiotic-use/viral-infections.html",
            trust=0.90,
            stance="entails",
        )
    ]
    coverage = compute_subclaim_coverage(
        subclaims=[claim],
        evidence_list=evidence,
        anchors_by_subclaim=anchors.anchors_by_subclaim,
    )
    assert coverage["details"][0]["status"] == "STRONGLY_VALID"
