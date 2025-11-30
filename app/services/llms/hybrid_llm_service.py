from typing import Any, Dict, Literal

from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService
from app.services.llms.ollama_service import OllamaService

logger = get_logger(__name__)


class HybridLLMService:
    """
    Hybrid LLM service that uses local Ollama as primary with Groq as fallback.

    Strategy:
    - First, try to get response from local Ollama
    - Extract confidence score from Ollama response
    - If confidence < threshold, fallback to Groq (cloud-based, more reliable)
    - Return the better response with metadata about which service was used
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        ollama_host: str = "ollama",
        ollama_port: int = 11434,
        ollama_model: str = "mistral",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.groq_service = GroqService()
        self.ollama_service = OllamaService(
            host=ollama_host,
            port=ollama_port,
            model=ollama_model,
        )
        logger.info(f"[HybridLLMService] Initialized with confidence_threshold={confidence_threshold}")

    async def ainvoke(
        self,
        prompt: str,
        response_format: Literal["text", "json"] = "text",
        prefer_local: bool = True,
        use_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Invoke LLM with confidence-based fallback strategy.

        Args:
            prompt: The input prompt
            response_format: "text" or "json" output format
            prefer_local: If True, try Ollama first; if False, use Groq directly
            use_fallback: If True, fallback to Groq when Ollama confidence is low

        Returns:
            {
                "text": "...",  # or "json": {...} if response_format="json"
                "confidence": 0.85,
                "source": "ollama",  # or "groq"
                "ollama_confidence": 0.65,  # if fallback occurred
            }
        """
        result: Dict[str, Any] = {"source": "unknown"}

        if not prefer_local:
            # Skip Ollama, use Groq directly
            logger.info("[HybridLLMService] Skipping Ollama, using Groq directly")
            groq_response = await self.groq_service.ainvoke(prompt, response_format)
            groq_response["source"] = "groq"
            groq_response["confidence"] = 1.0  # Groq responses are typically high-confidence
            return groq_response

        # Try Ollama first
        try:
            logger.info("[HybridLLMService] Attempting local inference with Ollama...")
            ollama_response = await self.ollama_service.ainvoke(
                prompt,
                response_format,
                return_confidence=True,
            )
            ollama_confidence = ollama_response.get("confidence", 0.0)
            result["ollama_confidence"] = ollama_confidence
            result["source"] = "ollama"

            logger.info(f"[HybridLLMService] Ollama response received with confidence={ollama_confidence:.2f}")

            # Check if we should fallback to Groq
            if use_fallback and ollama_confidence < self.confidence_threshold:
                logger.warning(
                    f"[HybridLLMService] Ollama confidence {ollama_confidence:.2f} < "
                    f"threshold {self.confidence_threshold}, falling back to Groq..."
                )
                groq_response = await self.groq_service.ainvoke(prompt, response_format)
                groq_response["source"] = "groq"
                groq_response["confidence"] = 1.0
                groq_response["ollama_confidence"] = ollama_confidence
                return groq_response

            # Ollama response is good enough
            return ollama_response

        except Exception as e:
            logger.warning(f"[HybridLLMService] Ollama failed: {e}")

            if not use_fallback:
                raise

            # Fallback to Groq on any Ollama error
            logger.info("[HybridLLMService] Falling back to Groq due to Ollama error...")
            try:
                groq_response = await self.groq_service.ainvoke(prompt, response_format)
                groq_response["source"] = "groq"
                groq_response["confidence"] = 1.0
                groq_response["ollama_error"] = str(e)
                return groq_response
            except Exception as groq_error:
                logger.error(f"[HybridLLMService] Both Ollama and Groq failed. " f"Ollama: {e}, Groq: {groq_error}")
                raise RuntimeError(f"Both Ollama and Groq failed. Ollama: {e}, Groq: {groq_error}")

    async def set_confidence_threshold(self, threshold: float) -> None:
        """Dynamically adjust confidence threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self.confidence_threshold = threshold
        logger.info(f"[HybridLLMService] Confidence threshold updated to {threshold}")

    async def health_check(self) -> Dict[str, Any]:
        """Check health of both services."""
        ollama_healthy = await self.ollama_service._health_check()
        groq_healthy = bool(self.groq_service.client)

        return {
            "ollama": ollama_healthy,
            "groq": groq_healthy,
            "hybrid_ready": ollama_healthy or groq_healthy,
            "confidence_threshold": self.confidence_threshold,
        }
