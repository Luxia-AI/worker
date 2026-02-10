import asyncio
import json
from typing import Any, Dict

from groq import AsyncGroq

from app.constants.config import LLM_MAX_TOKENS_DEFAULT, LLM_MODEL_NAME, LLM_TEMPERATURE
from app.core.config import settings
from app.core.logger import get_logger
from app.core.rate_limit import throttled

logger = get_logger(__name__)


# Custom exception for rate limiting
class RateLimitError(Exception):
    """Raised when Groq API returns 429 rate limit error"""

    pass


class GroqService:
    def __init__(self) -> None:
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")

        self.client = AsyncGroq(api_key=api_key)
        self.model = LLM_MODEL_NAME

    @throttled(limit=10, period=60.0, name="groq_api")
    async def ainvoke(
        self,
        prompt: str,
        response_format: str = "text",
        max_retries: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """
        Calls Groq async chat completion endpoint with retry logic for rate limits.
        Supports JSON or text output.

        Args:
            prompt: The prompt to send to the LLM
            response_format: "json" or "text" response format
            max_retries: Maximum retry attempts on rate limit (default 1, used by hybrid service)
            temperature: Override default temperature (0.0-1.0, lower = more deterministic)

        Raises:
            RateLimitError: When rate limited after max retries
            Exception: On other API errors
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
            "max_tokens": int(max_tokens if max_tokens is not None else LLM_MAX_TOKENS_DEFAULT),
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message

                # JSON response
                if response_format == "json":
                    if msg.content:
                        result: Dict[str, Any] = json.loads(msg.content)
                        return result
                    return {}

                # Text response
                return {"text": msg.content}

            except Exception as e:
                error_str = str(e)

                # Check for rate limit error
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(
                            f"[GroqService] Rate limit hit. Retrying in {wait_time}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"[GroqService] Rate limit exceeded after {max_retries} attempts: {e}")
                        raise RateLimitError(f"Groq rate limit exceeded: {e}") from e
                else:
                    logger.error(f"[GroqService] Groq call failed: {e}")
                    raise

        raise RuntimeError("Unexpected error in Groq service")
