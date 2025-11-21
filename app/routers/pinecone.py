from typing import Any, Dict

from fastapi import APIRouter

from app.core.logger import get_logger
from app.services.vdb.vdb_retrieval import VDBRetrieval

logger = get_logger(__name__)

router = APIRouter()


@router.get("/search")
async def test_pinecone(query: str = "Who founded SpaceX?") -> Dict[str, Any]:
    logger.info(f"Search request received: {query}")
    try:
        vdb_retrieval = VDBRetrieval()
        results = await vdb_retrieval.search(query, top_k=3)
        logger.info(f"Search completed, returned {len(results)} results")
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"query": query, "error": str(e), "results": []}
