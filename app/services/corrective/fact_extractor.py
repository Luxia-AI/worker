import json
from typing import Any, Dict

from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService

logger = get_logger(__name__)


class FactExtractingLLM:
    """
    Async wrapper for Groq API using MoonshotAI's kimi-k2-instruct model.
    Uses OpenAI-compatible Chat Completions API.
    """

    def __init__(self) -> None:
        self.groq_service = GroqService()

    async def ainvoke(self, prompt: str, response_format: str = "text") -> Dict[str, Any]:
        """
        Calls Groq async chat completion endpoint.
        Supports JSON or text output.
        """

        kwargs: Dict[str, Any] = {
            "model": self.groq_service.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        # Use Groq-supported JSON response format
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self.groq_service.client.chat.completions.create(**kwargs)
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
            logger.error(f"[FactExtractingLLM] Groq call failed: {e}")
            raise
