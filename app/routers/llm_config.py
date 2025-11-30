"""
LLM Configuration and Health Check Routes

Endpoints:
  GET /llm/health - Health status of LLM services
  GET /llm/config - Current LLM configuration
  POST /llm/config - Update LLM configuration
  POST /llm/test - Test LLM with a sample prompt
"""

from typing import Any, Dict, Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import get_logger
from app.services.llms.factory import get_hybrid_llm_service, get_llm_service

logger = get_logger(__name__)

router = APIRouter()


class LLMConfigUpdate(BaseModel):
    """Request model for updating LLM configuration."""

    service_type: Literal["groq", "ollama", "hybrid"] | None = None
    confidence_threshold: float | None = None


class TestPromptRequest(BaseModel):
    """Request model for testing LLM."""

    prompt: str
    response_format: Literal["text", "json"] = "text"


@router.get("/llm/health")
async def llm_health() -> Dict[str, Any]:
    """
    Check health status of all LLM services.

    Returns:
        {
            "ollama": true/false,
            "groq": true/false,
            "hybrid_ready": true/false,
            "confidence_threshold": 0.7,
            "service_type": "hybrid"
        }
    """
    try:
        hybrid_service = get_hybrid_llm_service()
        health = await hybrid_service.health_check()
        health["service_type"] = settings.LLM_SERVICE_TYPE
        return health
    except Exception as e:
        logger.error(f"[LLM Health] Error checking health: {e}")
        return {
            "error": str(e),
            "service_type": settings.LLM_SERVICE_TYPE,
            "ollama": False,
            "groq": False,
            "hybrid_ready": False,
        }


@router.get("/llm/config")
async def get_llm_config() -> Dict[str, Any]:
    """
    Get current LLM configuration.

    Returns:
        {
            "service_type": "hybrid",
            "confidence_threshold": 0.7,
            "ollama_host": "ollama",
            "ollama_port": 11434,
            "ollama_model": "mistral"
        }
    """
    return {
        "service_type": settings.LLM_SERVICE_TYPE,
        "confidence_threshold": settings.LLM_CONFIDENCE_THRESHOLD,
        "ollama_host": settings.OLLAMA_HOST,
        "ollama_port": settings.OLLAMA_PORT,
        "ollama_model": settings.OLLAMA_MODEL,
    }


@router.post("/llm/config")
async def update_llm_config(update: LLMConfigUpdate) -> Dict[str, Any]:
    """
    Update LLM configuration.

    Only confidence_threshold can be updated at runtime.
    Other settings require app restart.

    Returns:
        Updated configuration
    """
    try:
        if update.confidence_threshold is not None:
            if not 0.0 <= update.confidence_threshold <= 1.0:
                return {"error": "confidence_threshold must be between 0.0 and 1.0"}

            hybrid_service = get_hybrid_llm_service()
            await hybrid_service.set_confidence_threshold(update.confidence_threshold)
            logger.info(f"[LLM Config] Updated confidence_threshold to " f"{update.confidence_threshold}")

        if update.service_type is not None:
            logger.warning(
                f"[LLM Config] service_type change requested ({update.service_type}) "
                "but requires app restart. Change LLM_SERVICE_TYPE env var and restart."
            )

        return {
            "service_type": settings.LLM_SERVICE_TYPE,
            "confidence_threshold": (
                update.confidence_threshold
                if update.confidence_threshold is not None
                else settings.LLM_CONFIDENCE_THRESHOLD
            ),
            "message": ("confidence_threshold updated; " "service_type changes require app restart"),
        }

    except Exception as e:
        logger.error(f"[LLM Config] Error updating config: {e}")
        return {"error": str(e)}


@router.post("/llm/test")
async def test_llm(request: TestPromptRequest) -> Dict[str, Any]:
    """
    Test LLM with a sample prompt.

    Returns response and metadata about which service was used.

    Example:
        POST /llm/test
        {
            "prompt": "What is 2+2?",
            "response_format": "text"
        }
    """
    try:
        llm_service = get_llm_service(
            service_type=settings.LLM_SERVICE_TYPE,
            confidence_threshold=settings.LLM_CONFIDENCE_THRESHOLD,
        )

        response = await llm_service.ainvoke(
            request.prompt,
            response_format=request.response_format,
        )

        logger.info(
            f"[LLM Test] Success using {response.get('source', 'unknown')} "
            f"with confidence={response.get('confidence', 'N/A')}"
        )

        return response

    except Exception as e:
        logger.error(f"[LLM Test] Error testing LLM: {e}")
        return {"error": str(e), "source": "error"}
