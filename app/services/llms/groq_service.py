import asyncio
import json
import os
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
        primary_api_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
        fallback_api_key = os.getenv("GROQ_API_KEY_FALLBACK") or settings.GROQ_API_KEY_FALLBACK
        if not primary_api_key and not fallback_api_key:
            raise RuntimeError("Missing GROQ_API_KEY")
        self.model = LLM_MODEL_NAME
        self._clients: list[tuple[str, AsyncGroq]] = []

        primary_headers = self._build_headers(
            os.getenv("GROQ_ORG_ID") or settings.GROQ_ORG_ID,
            os.getenv("GROQ_PROJECT_ID") or settings.GROQ_PROJECT_ID,
        )
        if primary_api_key:
            self._clients.append(("primary", AsyncGroq(api_key=primary_api_key, default_headers=primary_headers)))

        fallback_headers = self._build_headers(
            os.getenv("GROQ_ORG_ID_FALLBACK") or settings.GROQ_ORG_ID_FALLBACK,
            os.getenv("GROQ_PROJECT_ID_FALLBACK") or settings.GROQ_PROJECT_ID_FALLBACK,
        )
        if fallback_api_key and fallback_api_key != primary_api_key:
            self._clients.append(("fallback", AsyncGroq(api_key=fallback_api_key, default_headers=fallback_headers)))

        if not self._clients:
            raise RuntimeError("Missing Groq client credentials")

    @property
    def has_fallback_client(self) -> bool:
        return any(name == "fallback" for name, _ in self._clients)

    @staticmethod
    def _build_headers(org_id: str | None, project_id: str | None) -> Dict[str, str] | None:
        headers: Dict[str, str] = {}
        if org_id:
            headers["x-groq-organization"] = org_id
            headers["groq-organization"] = org_id
        if project_id:
            headers["x-groq-project"] = project_id
            headers["groq-project"] = project_id
        return headers or None

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        error_str = str(error or "")
        return "429" in error_str or "rate_limit" in error_str.lower()

    @throttled(limit=10, period=60.0, name="groq_api")
    async def ainvoke(
        self,
        prompt: str,
        response_format: str = "text",
        max_retries: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        force_client: str | None = None,
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

        clients_to_use = self._clients
        if force_client:
            requested = force_client.strip().lower()
            clients_to_use = [(name, client) for name, client in self._clients if name == requested]
            if not clients_to_use:
                raise RuntimeError(f"Requested Groq client '{force_client}' is not configured")

        last_error: Exception | None = None
        for client_idx, (client_name, client) in enumerate(clients_to_use):
            is_last_client = client_idx == len(clients_to_use) - 1
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(**kwargs)
                    msg = response.choices[0].message

                    # JSON response
                    if response_format == "json":
                        if msg.content:
                            result: Dict[str, Any] = json.loads(msg.content)
                            logger.info("[GroqService] Request served by %s Groq credentials", client_name)
                            return result
                        logger.info("[GroqService] Request served by %s Groq credentials", client_name)
                        return {}

                    # Text response
                    logger.info("[GroqService] Request served by %s Groq credentials", client_name)
                    return {"text": msg.content}

                except Exception as e:
                    last_error = e
                    if self._is_rate_limit_error(e):
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                            logger.warning(
                                f"[GroqService] Rate limit hit on {client_name} client. Retrying in {wait_time}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        logger.error(
                            "[GroqService] Rate limit exceeded on %s client after %d attempts: %s",
                            client_name,
                            max_retries,
                            e,
                        )
                        if not is_last_client:
                            logger.warning("[GroqService] Switching to fallback Groq credentials after rate limits")
                        break

                    logger.error(f"[GroqService] Groq call failed on {client_name} client: {e}")
                    if not is_last_client:
                        logger.warning("[GroqService] Switching to fallback Groq credentials after failure")
                    break

            if not is_last_client:
                continue

        if isinstance(last_error, Exception) and self._is_rate_limit_error(last_error):
            raise RateLimitError(f"Groq rate limit exceeded: {last_error}") from last_error
        if isinstance(last_error, Exception):
            raise last_error

        raise RuntimeError("Unexpected error in Groq service")
