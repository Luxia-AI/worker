"""
Hybrid LLM Service - Groq with Ollama fallback
"""

from typing import Any, Dict, Optional

from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService, RateLimitError
from app.services.llms.ollama_service import OllamaService

logger = get_logger(__name__)


class HybridLLMService:
    """
    Hybrid LLM service that tries Groq first and falls back to Ollama
    if Groq fails (rate limit, API error, etc).
    """

    groq_service: Optional[GroqService]
    ollama_service: Optional[OllamaService]

    def __init__(self) -> None:
        try:
            self.groq_service = GroqService()
            self.groq_available = True
        except Exception as e:
            logger.warning(f"[HybridLLMService] Groq service unavailable: {e}")
            self.groq_service = None
            self.groq_available = False

        try:
            self.ollama_service = OllamaService()
            self.ollama_available = True
        except Exception as e:
            logger.warning(f"[HybridLLMService] Ollama service unavailable: {e}")
            self.ollama_service = None
            self.ollama_available = False

        if not self.groq_available and not self.ollama_available:
            raise RuntimeError("Neither Groq nor Ollama LLM services are available")

    async def ainvoke(self, prompt: str, response_format: str = "text") -> Dict[str, Any]:
        """
        Calls LLM with automatic fallback.
        Tries Groq up to 3 times, then falls back to Ollama on failure.
        """

        # Try Groq up to 3 times if available
        if self.groq_available and self.groq_service:
            for attempt in range(3):
                try:
                    logger.debug(f"[HybridLLMService] Attempting Groq (attempt {attempt + 1}/3)...")
                    result = await self.groq_service.ainvoke(prompt, response_format, max_retries=1)
                    logger.debug("[HybridLLMService] Groq succeeded")
                    return result
                except RateLimitError as e:
                    if attempt < 2:  # Try 2 more times
                        logger.warning(
                            f"[HybridLLMService] Groq rate limited (attempt {attempt + 1}/3): {e}. Retrying..."
                        )
                        continue
                    else:
                        logger.warning(
                            f"[HybridLLMService] Groq rate limited after 3 attempts: {e}. Falling back to Ollama..."
                        )
                        break
                except Exception as e:
                    logger.warning(f"[HybridLLMService] Groq failed (attempt {attempt + 1}/3): {e}. Retrying...")
                    if attempt < 2:
                        continue
                    else:
                        logger.warning(
                            f"[HybridLLMService] Groq failed after 3 attempts: {e}. Falling back to Ollama..."
                        )
                        break

        # Fallback to Ollama
        if self.ollama_available and self.ollama_service:
            try:
                logger.info("[HybridLLMService] Using Ollama...")
                result = await self.ollama_service.ainvoke(prompt, response_format)
                logger.info("[HybridLLMService] Ollama succeeded")
                return result
            except Exception as e:
                logger.error(f"[HybridLLMService] Ollama failed: {e}")
                raise

        raise RuntimeError("No LLM service available")
