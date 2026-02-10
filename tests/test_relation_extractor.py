import pytest

from app.services.corrective.relation_extractor import RelationExtractor


def _make_extractor() -> RelationExtractor:
    # Avoid __init__ (which spins up LLM clients) for deterministic unit tests.
    return RelationExtractor.__new__(RelationExtractor)


@pytest.mark.asyncio
async def test_relation_extractor():
    ex = RelationExtractor()
    facts = [
        {
            "fact_id": "f1",
            "statement": "COVID-19 vaccines reduce hospitalization by preventing severe disease.",
            "source_url": "https://who.int/example",
        },
        {
            "fact_id": "f2",
            "statement": "Ibuprofen may increase the risk of liver damage in people with preexisting liver disease.",
            "source_url": "https://example.org/article",
        },
    ]
    entities = ["covid-19", "vaccines", "hospitalization", "ibuprofen", "liver damage", "liver disease"]

    triples = await ex.extract_relations(facts, entities)
    for t in triples:
        print(t)


def test_negation_guard_inverts_positive_causal_relation():
    ex = _make_extractor()
    statement = "Large studies show vaccines do not cause autism."
    relation = ex._apply_negation_guard(statement, "causes")
    assert relation == "does_not_cause"


def test_negation_guard_drops_unmapped_negated_relation():
    ex = _make_extractor()
    statement = "There is no evidence of a causal link between vaccines and autism."
    relation = ex._apply_negation_guard(statement, "induces pathology in")
    assert relation is None


def test_negation_guard_maps_contributes_to_into_negative_causal_relation():
    ex = _make_extractor()
    statement = "Large studies show vaccines do not cause autism."
    relation = ex._apply_negation_guard(statement, "contributes to")
    assert relation == "does_not_cause"
