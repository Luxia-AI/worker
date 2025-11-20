import json
from typing import Any, Dict

from groq import AsyncGroq

from app.constants.config import LLM_MODEL_NAME, LLM_TEMPERATURE
from app.core.config import settings
from app.core.logger import get_logger
from app.core.rate_limit import throttled

logger = get_logger(__name__)


class GroqService:
    def __init__(self) -> None:
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")

        self.client = AsyncGroq(api_key=api_key)

        # MoonshotAI model
        self.model = LLM_MODEL_NAME

    @throttled(limit=10, period=60.0, name="groq_api")
    async def ainvoke(self, prompt: str, response_format: str = "text") -> Dict[str, Any]:
        """
        Calls Groq async chat completion endpoint.
        Supports JSON or text output.
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
            logger.error(f"[GroqService] Groq call failed: {e}")
            raise
