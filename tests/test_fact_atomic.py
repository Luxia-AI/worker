from app.services.corrective.fact_extractor import FactExtractor


def test_atomic_split_on_semicolon():
    fe = FactExtractor.__new__(FactExtractor)
    facts = [
        {
            "statement": "The adult human skeleton has 206 bones; 106 are in the hands and feet.",
            "source_url": "https://example.com",
        }
    ]
    normalized = fe._normalize_atomic_facts(facts)
    assert len(normalized) == 2  # nosec
