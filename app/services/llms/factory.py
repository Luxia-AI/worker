"""
LLM Service Factory

Provides a centralized way to get LLM instances (Groq, Ollama, or Hybrid).
This allows easy switching between implementations and configuration management.
"""

from typing import Literal

from app.core.config import settings
from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService
from app.services.llms.hybrid_llm_service import HybridLLMService
from app.services.llms.ollama_service import OllamaService

logger = get_logger(__name__)

# Global service instances (lazy loaded)
_groq_service: GroqService | None = None
_ollama_service: OllamaService | None = None
_hybrid_service: HybridLLMService | None = None


def get_groq_service() -> GroqService:
    """Get or create Groq service instance."""
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqService()
        logger.info("[LLM Factory] Groq service initialized")
    return _groq_service


def get_ollama_service() -> OllamaService:
    """Get or create Ollama service instance."""
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaService()
        logger.info("[LLM Factory] Ollama service initialized")
    return _ollama_service


def get_hybrid_llm_service(
    confidence_threshold: float = 0.7,
) -> HybridLLMService:
    """Get or create Hybrid LLM service instance."""
    global _hybrid_service
    if _hybrid_service is None:
        _hybrid_service = HybridLLMService(
            confidence_threshold=confidence_threshold,
        )
        logger.info(
            f"[LLM Factory] Hybrid LLM service initialized with " f"confidence_threshold={confidence_threshold}"
        )
    return _hybrid_service


def get_llm_service(
    service_type: Literal["groq", "ollama", "hybrid"] = "hybrid",
    confidence_threshold: float = 0.7,
):
    """
    Get LLM service based on configuration.

    Args:
        service_type: Which service to use
        confidence_threshold: For hybrid service, minimum confidence threshold

    Returns:
        LLM service instance (GroqService, OllamaService, or HybridLLMService)
    """
    if service_type == "groq":
        return get_groq_service()
    elif service_type == "ollama":
        return get_ollama_service()
    elif service_type == "hybrid":
        return get_hybrid_llm_service(confidence_threshold)
    else:
        raise ValueError(f"Unknown service_type: {service_type}. " "Must be one of: 'groq', 'ollama', 'hybrid'")


# Environment-based configuration
LLM_SERVICE_TYPE: Literal["groq", "ollama", "hybrid"] = getattr(settings, "LLM_SERVICE_TYPE", "hybrid")
LLM_CONFIDENCE_THRESHOLD: float = getattr(settings, "LLM_CONFIDENCE_THRESHOLD", 0.7)

logger.info(f"[LLM Factory] Using service_type={LLM_SERVICE_TYPE}, " f"confidence_threshold={LLM_CONFIDENCE_THRESHOLD}")
