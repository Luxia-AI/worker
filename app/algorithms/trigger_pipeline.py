from typing import Any, Dict

from app.services.corrective.pipeline import CorrectivePipeline


async def run_worker_pipeline(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wraps the RAG pipeline and returns normalized output.
    """
    pipeline = CorrectivePipeline()

    # Extract text and domain from post
    post_text = post.get("text", "")
    domain = post.get("meta", {}).get("domain", "general")

    result = await pipeline.run(post_text, domain)

    return {
        "post_id": post["post_id"],
        "truth_score": result.get("truth_score", 0.0),
        "confidence": result.get("confidence", 0.0),
        "evidence": result.get("evidence", []),
        "sources": result.get("sources", []),
        "status": "completed",
    }
