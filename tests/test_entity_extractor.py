import pytest

from app.services.corrective.entity_extractor import EntityExtractor


@pytest.mark.asyncio
async def test_entity_extractor():
    ex = EntityExtractor()

    facts = [
        {"fact_id": "f1", "statement": "Vitamin D reduces the risk of bone fractures.", "entities": []},
        {"fact_id": "f2", "statement": "COVID-19 vaccines reduce hospitalization.", "entities": []},
    ]

    out = await ex.annotate_entities(facts)
    print(out)
