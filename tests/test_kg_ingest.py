import pytest

from app.services.kg.kg_ingest import KGIngest


@pytest.mark.asyncio
async def test_ingest_triples():
    ingest = KGIngest()
    triples = [
        {
            "subject": "vitamin d",
            "relation": "reduces",
            "object": "fracture risk",
            "confidence": 0.92,
            "source_url": "https://nih.gov/test",
            "published_at": "2024-01-01T00:00:00Z",
        }
    ]
    res = await ingest.ingest_triples(triples)
    print("inserted:", res)
