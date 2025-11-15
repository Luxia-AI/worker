from typing import Any, Dict

from fastapi import APIRouter

from app.services.retrieval import retrieve_similar

router = APIRouter()


@router.get("/search")
async def test_pinecone(query: str = "Who founded SpaceX?") -> Dict[str, Any]:
    results = retrieve_similar(query, top_k=3)
    return {"query": query, "results": results}
