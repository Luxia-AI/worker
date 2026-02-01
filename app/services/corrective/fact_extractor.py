import json
from typing import Any, Dict, List

from app.constants.llm_prompts import FACT_EXTRACTION_PROMPT
from app.core.logger import get_logger
from app.services.common.text_cleaner import clean_statement, truncate_content
from app.services.llms.hybrid_service import HybridLLMService

logger = get_logger(__name__)


class FactExtractor:
    """
    Async wrapper for Groq API (with Ollama fallback).
    Uses hybrid LLM service for fact extraction.
    """

    def __init__(self) -> None:
        self.llm_service = HybridLLMService()

    async def ainvoke(self, prompt: str, response_format: str = "text") -> Dict[str, Any]:
        """
        Calls LLM (Groq with Ollama fallback).
        Supports JSON or text output.
        """
        try:
            result = await self.llm_service.ainvoke(prompt, response_format)
            return result

        except Exception as e:
            logger.error(f"[FactExtractor] LLM call failed: {e}")
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

                # Validate extracted is a list
                if not isinstance(extracted, list):
                    logger.warning(f"[FactExtractor] Expected list of facts, got {type(extracted)}")
                    continue

                # Enrich with source information and clean statements
                for fact in extracted:
                    if not isinstance(fact, dict):
                        continue
                    fact["statement"] = clean_statement(fact.get("statement", ""))
                    fact["source_url"] = page.get("url", "")
                    fact["source"] = page.get("source", "")
                    fact["published_at"] = page.get("published_at", "")
                    fact["fact_id"] = f"f_{len(facts)}"

                facts.extend(extracted)
            except (KeyError, json.JSONDecodeError, TypeError) as e:
                logger.warning(f"[FactExtractor] Failed to extract from {page.get('url')}: {e}")
                continue

        logger.info(f"[FactExtractor] Extracted {len(facts)} facts from {len(scraped_pages)} pages")
        return facts
