from groq import AsyncGroq

from app.core.config import settings


class GroqService:
    def __init__(self) -> None:
        api_key = settings.groq_api_key
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")

        self.client = AsyncGroq(api_key=api_key)

        # MoonshotAI model
        self.model = "moonshotai/kimi-k2-instruct"
