import pytest

from app.services.corrective.relation_extractor import RelationExtractor


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
