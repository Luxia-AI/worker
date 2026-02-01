"""
Ollama LLM Service - Local LLM fallback
"""

import json
import os
from typing import Any, Dict

import httpx

from app.constants.config import LLM_TEMPERATURE
from app.core.logger import get_logger

logger = get_logger(__name__)


class OllamaService:
    """Local Ollama LLM service as a fallback."""

    def __init__(self) -> None:
        self.host = os.getenv("OLLAMA_HOST", "ollama")
        self.port = os.getenv("OLLAMA_PORT", "11434")
        self.model = os.getenv("OLLAMA_MODEL", "tinyllama")
        self.base_url = f"http://{self.host}:{self.port}"

    async def ainvoke(self, prompt: str, response_format: str = "text") -> Dict[str, Any]:
        """
        Calls Ollama local LLM.
        Supports JSON or text output.
        Short timeout to fail fast if Ollama is not available.
        """
        try:
            # Short timeout: 5s connect, 30s read - fail fast if Ollama not available
            timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": LLM_TEMPERATURE,
                    },
                )

                if response.status_code != 200:
                    raise Exception(f"Ollama error: {response.status_code}")

                result = response.json()
                text = result.get("response", "").strip()

                # JSON response
                if response_format == "json":
                    if text:
                        try:
                            parsed = json.loads(text)
                            return parsed
                        except json.JSONDecodeError:
                            logger.warning(f"[OllamaService] Failed to parse JSON response: {text[:100]}")
                            return {}
                    return {}

                # Text response
                return {"text": text}

        except Exception as e:
            logger.error(f"[OllamaService] Ollama call failed: {e}")
            raise
