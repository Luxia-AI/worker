from typing import Any, Dict

from fastapi import APIRouter

from app.core.logger import get_logger
from app.services.retrieval import retrieve_similar

logger = get_logger(__name__)

router = APIRouter()


@router.get("/search")
async def test_pinecone(query: str = "Who founded SpaceX?") -> Dict[str, Any]:
    logger.info(f"Search request received: {query}")
    results = retrieve_similar(query, top_k=3)
    logger.info(f"Search completed, returned {len(results)} results")
    return {"query": query, "results": results}
