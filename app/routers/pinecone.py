from fastapi import APIRouter
from app.services.upsert import insert_dummy_data
from app.services.retrieval import retrieve_similar

router = APIRouter()

@router.get("/search")
async def test_pinecone(query: str = "Who founded SpaceX?"):
    results = retrieve_similar(query, top_k=3)
    return {
        "query": query,
        "results": results
    }