import asyncio
import json
import re
from typing import Any, Dict

from groq import AsyncGroq

from app.constants.config import LLM_MODEL_NAME, LLM_TEMPERATURE
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class GroqService:
    def __init__(self) -> None:
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")

        self.client = AsyncGroq(api_key=api_key)

        # MoonshotAI model
        self.model = LLM_MODEL_NAME

        # Rate limit retry configuration
        self.max_retries = 5
        self.base_backoff = 1.0  # Start with 1 second
        self.max_backoff = 60.0  # Cap at 60 seconds

    def _extract_retry_after(self, error_msg: str) -> float | None:
        """Extract retry-after time from error message if available."""
        # Try to find "Please try again in X.XXXs"
        match = re.search(r"Please try again in ([0-9.]+)s", error_msg)
        if match:
            return float(match.group(1))
        return None

    async def ainvoke(self, prompt: str, response_format: str = "text", retry_count: int = 0) -> Dict[str, Any]:
        """
        Calls Groq async chat completion endpoint with exponential backoff retry.
        Supports JSON or text output.
        Automatically retries on rate limit errors (429).
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
        }

        # Use Groq-supported JSON response format
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

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

            # Check if it's a rate limit error (429)
            if "429" in error_str and retry_count < self.max_retries:
                # Extract suggested retry-after time if available
                retry_after = self._extract_retry_after(error_str)

                if retry_after:
                    wait_time = min(retry_after, self.max_backoff)
                    logger.warning(
                        f"[GroqService] Rate limit hit. Retrying in {wait_time:.1f}s "
                        f"(attempt {retry_count + 1}/{self.max_retries})"
                    )
                else:
                    # Exponential backoff: base_backoff * 2^retry_count
                    wait_time = min(self.base_backoff * (2**retry_count), self.max_backoff)
                    logger.warning(
                        f"[GroqService] Rate limit hit. Retrying in {wait_time:.1f}s "
                        f"(attempt {retry_count + 1}/{self.max_retries})"
                    )

                await asyncio.sleep(wait_time)
                # Recursive retry with incremented count
                return await self.ainvoke(prompt, response_format, retry_count + 1)

            # For non-rate-limit errors or max retries exceeded
            logger.error(f"[GroqService] Groq call failed: {e}")
            raise
