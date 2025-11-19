import pytest

from app.services.corrective.fact_extractor import FactExtractingLLM


@pytest.mark.asyncio
async def test_fact_extracting():
    llm = FactExtractingLLM()
    print("Testing text mode:")
    r1 = await llm.ainvoke("Say hello briefly", response_format="text")
    print(r1)

    print("\nTesting JSON mode:")
    prompt = """
    Return ONLY valid JSON:
    {"message": "hello", "value": 123}
    """
    r2 = await llm.ainvoke(prompt, response_format="json")
    print(r2)
