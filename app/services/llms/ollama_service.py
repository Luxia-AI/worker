import asyncio
import json
import re
from typing import Any, Dict

import aiohttp

from app.constants.config import LLM_TEMPERATURE
from app.core.logger import get_logger

logger = get_logger(__name__)


class OllamaService:
    """
    Local LLM service using Ollama.
    Supports confidence-based responses and can extract confidence scores.
    """

    def __init__(self, host: str = "ollama", port: int = 11434, model: str = "mistral") -> None:
        self.host = host
        self.port = port
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for local LLM

    async def _health_check(self) -> bool:
        """Check if Ollama service is healthy."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as resp:
                    return resp.status == 200
        except Exception as e:
            logger.warning(f"[OllamaService] Health check failed: {e}")
            return False

    async def _extract_confidence(self, response_text: str) -> float:
        """
        Extract confidence score from response.
        Heuristics:
        - Look for explicit confidence markers like "confidence: 0.85"
        - Look for certainty keywords (very certain, certain, uncertain, etc.)
        - Default to length-based heuristic if model generates verbose response
        """
        # Explicit confidence patterns
        confidence_patterns = [
            r"confidence[:\s]+([0-9.]+)",
            r"confidence[:\s]+([0-9]+%)",
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                conf_val = match.group(1)
                # Handle percentage
                if "%" in conf_val:
                    return float(conf_val.rstrip("%")) / 100.0
                return float(conf_val)

        # Keyword-based heuristics
        certainty_keywords = {
            "very certain": 0.95,
            "certain": 0.85,
            "fairly confident": 0.75,
            "somewhat confident": 0.65,
            "uncertain": 0.45,
            "very uncertain": 0.25,
        }

        text_lower = response_text.lower()
        for keyword, score in certainty_keywords.items():
            if keyword in text_lower:
                return score

        # Length-based heuristic: longer, more detailed responses tend to be more confident
        word_count = len(response_text.split())
        if word_count > 200:
            return min(0.85, 0.5 + (word_count / 1000))
        elif word_count > 100:
            return min(0.80, 0.5 + (word_count / 500))
        else:
            return min(0.65, 0.4 + (word_count / 250))

    async def ainvoke(
        self,
        prompt: str,
        response_format: str = "text",
        return_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Call Ollama local model with optional confidence extraction.

        Args:
            prompt: The input prompt
            response_format: "text" or "json" (json requires model to support it)
            return_confidence: Whether to extract and return confidence score

        Returns:
            For text: {"text": "...", "confidence": 0.85} if return_confidence=True
            For json: {"json": {...}, "confidence": 0.85} if return_confidence=True
        """
        health_ok = await self._health_check()
        if not health_ok:
            logger.error("[OllamaService] Ollama service is not healthy")
            raise RuntimeError("Ollama service unavailable")

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": LLM_TEMPERATURE,
                    "stream": False,
                }

                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Ollama API returned {resp.status}")

                    response_data = await resp.json()
                    text_response = response_data.get("response", "")

                    if not text_response:
                        logger.warning("[OllamaService] Empty response from Ollama")
                        return (
                            {
                                "text": "",
                                "confidence": 0.0,
                            }
                            if return_confidence
                            else {"text": ""}
                        )

                    # Extract confidence if requested
                    result: Dict[str, Any] = {}
                    if response_format == "json":
                        try:
                            # Try to parse JSON from response
                            json_match = re.search(r"\{.*\}", text_response, re.DOTALL)
                            if json_match:
                                result["json"] = json.loads(json_match.group(0))
                            else:
                                result["json"] = {}
                        except json.JSONDecodeError:
                            logger.warning("[OllamaService] Failed to parse JSON from response")
                            result["json"] = {}
                    else:
                        result["text"] = text_response

                    if return_confidence:
                        confidence = await self._extract_confidence(text_response)
                        result["confidence"] = confidence
                        logger.debug(f"[OllamaService] Response confidence: {confidence:.2f}")

                    return result

        except asyncio.TimeoutError:
            logger.error("[OllamaService] Ollama request timed out")
            raise RuntimeError("Ollama request timeout")
        except Exception as e:
            logger.error(f"[OllamaService] Ollama call failed: {e}")
            raise
