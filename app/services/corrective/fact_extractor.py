import json
from typing import Any, Dict, List

from app.constants.llm_prompts import FACT_EXTRACTION_PROMPT
from app.core.logger import get_logger
from app.services.common.text_cleaner import clean_statement, truncate_content
from app.services.llms.groq_service import GroqService

logger = get_logger(__name__)


class FactExtractor:
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
            logger.error(f"[FactExtractor] Groq call failed: {e}")
            raise

    async def extract(self, scraped_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract facts from scraped web pages using Groq LLM.

        Args:
            scraped_pages: List of dicts with 'content', 'source', 'url', 'published_at'

        Returns:
            List of extracted facts with statement, confidence, source info
        """
        if not scraped_pages:
            return []

        facts: List[Dict[str, Any]] = []

        for page in scraped_pages:
            content = page.get("content", "")
            if not content or len(content) < 10:
                continue

            # Truncate long content to avoid token limits
            content_chunk = truncate_content(content, max_length=2000)

            prompt = FACT_EXTRACTION_PROMPT.format(content=content_chunk)

            try:
                result = await self.ainvoke(prompt, response_format="json")
                extracted = result.get("facts", [])

                # Enrich with source information and clean statements
                for fact in extracted:
                    fact["statement"] = clean_statement(fact.get("statement", ""))
                    fact["source_url"] = page.get("url", "")
                    fact["source"] = page.get("source", "")
                    fact["published_at"] = page.get("published_at", "")
                    fact["fact_id"] = f"f_{len(facts)}"

                facts.extend(extracted)
            except Exception as e:
                logger.warning(f"[FactExtractor] Failed to extract from {page.get('url')}: {e}")
                continue

        logger.info(f"[FactExtractor] Extracted {len(facts)} facts from {len(scraped_pages)} pages")
        return facts
