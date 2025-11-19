import json
from typing import Any, Dict

from groq import AsyncGroq

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class GroqService:
    def __init__(self) -> None:
        api_key = settings.groq_api_key
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")

        self.client = AsyncGroq(api_key=api_key)

        # MoonshotAI model
        self.model = "moonshotai/kimi-k2-instruct"

    async def ainvoke(self, prompt: str, response_format: str = "text") -> Dict[str, Any]:
        """
        Calls Groq async chat completion endpoint.
        Supports JSON or text output.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
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
                    return json.loads(msg.content)
                return {}

            # Text response
            return {"text": msg.content}

        except Exception as e:
            logger.error(f"[GroqService] Groq call failed: {e}")
            raise
