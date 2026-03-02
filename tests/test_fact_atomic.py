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


def test_ground_statement_to_source_preserves_negation_on_mismatch():
    fe = FactExtractor()
    source_text = (
        "Medicines must be approved by FDA before they can be sold or marketed. "
        "Supplements do not require this approval."
    )
    extracted = "Supplements require FDA approval before they can be sold or marketed."
    grounded = fe._ground_statement_to_source(extracted, source_text)
    assert "do not require" in grounded.lower()


def test_ground_statement_to_source_keeps_statement_when_no_strong_alignment():
    fe = FactExtractor()
    source_text = "Calcium and vitamin D help maintain bone health."
    extracted = "Omega-3 fatty acids reduce triglyceride levels in adults."
    grounded = fe._ground_statement_to_source(extracted, source_text)
    assert grounded == extracted
