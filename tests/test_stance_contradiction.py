from app.services.ranking.trust_ranker import DummyStanceClassifier, EvidenceItem, TrustRankingModule


def test_dummy_stance_classifier_marks_may_cause_vs_do_not_cause_as_contradiction():
    classifier = DummyStanceClassifier()
    claim = "Vaccines do not cause autism or the flu."
    evidence = "Some articles suggest vaccines may cause autism in children."
    assert classifier.classify_stance(claim, evidence) == "contradicts"


def test_module_classify_stance_for_evidence_updates_items_in_place():
    module = TrustRankingModule()
    evidence = [
        EvidenceItem(
            statement="Infant vaccines may cause autism.",
            semantic_score=0.7,
            source_url="https://example.com/a",
            stance="neutral",
            trust=0.7,
        ),
        EvidenceItem(
            statement="Vaccines do not cause autism.",
            semantic_score=0.8,
            source_url="https://example.com/b",
            stance="neutral",
            trust=0.8,
        ),
    ]

    module.classify_stance_for_evidence("Vaccines do not cause autism.", evidence)

    assert evidence[0].stance == "contradicts"
    assert evidence[1].stance == "entails"
