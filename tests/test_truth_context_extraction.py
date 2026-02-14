from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractor


def test_fact_truth_grounded_filter_rejects_hedged():
    extractor = FactExtractor()
    facts = [
        {"statement": "mRNA vaccines may integrate into DNA in rare cases.", "confidence": 0.8},
        {"statement": "mRNA does not enter the nucleus.", "confidence": 0.9},
    ]
    out = extractor._filter_truth_grounded_facts(facts)
    assert len(out) == 1
    assert "does not enter the nucleus" in out[0]["statement"].lower()


def test_entity_filter_removes_meta_terms():
    ents = ["Study", "research", "DNA", "mRNA vaccine", "claim"]
    out = EntityExtractor._filter_truth_entities(ents)
    assert "study" not in out
    assert "research" not in out
    assert "claim" not in out
    assert "dna" in out
    assert "mrna vaccine" in out
