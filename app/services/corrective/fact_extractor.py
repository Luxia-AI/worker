import json
from typing import Any, Dict, List, Optional

from app.constants.llm_prompts import FACT_EXTRACTION_PROMPT
from app.core.logger import get_logger
from app.services.common.text_cleaner import clean_statement, truncate_content
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)

# Batch fact extraction prompt - extract facts from multiple pages at once
BATCH_FACT_PROMPT = """Extract key factual statements from each content section below.
For each section, return the facts found.

IMPORTANT: You MUST respond with valid JSON only. No markdown, no explanations.
Return ONLY valid JSON with this exact structure:
{"results": [{"index": 0, "facts": [{"statement": "...", "confidence": 0.85}]}, {"index": 1, "facts": [...]}]}

SECTIONS:
"""

# Retry prompt when LLM returns non-dict
RETRY_JSON_PROMPT = """Your previous response was not valid JSON. Please respond ONLY with valid JSON.
No markdown code blocks, no explanations, just the JSON object.

Required format:
{required_format}

Original request:
{original_prompt}
"""


class FactExtractor:
    """
    Async wrapper for LLM-based fact extraction.
    Uses BATCH processing to minimize LLM calls and save Groq quota.
    """

    def __init__(self) -> None:
        self.llm_service = HybridLLMService()

    async def _parse_llm_response(
        self, result: Any, original_prompt: str, required_format: str, max_retries: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Parse LLM response, retry if non-dict returned."""
        # If already a valid dict with results, return it
        if isinstance(result, dict) and "results" in result:
            return result

        # If dict but missing results key, try to extract
        if isinstance(result, dict):
            # Maybe LLM returned {"facts": [...]} for single page
            if "facts" in result:
                return {"results": [{"index": 0, "facts": result["facts"]}]}
            logger.warning(f"[FactExtractor] Dict missing 'results' key: {list(result.keys())}")

        # If string, try to parse as JSON
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    if "results" in parsed:
                        return parsed
                    if "facts" in parsed:
                        return {"results": [{"index": 0, "facts": parsed["facts"]}]}
            except json.JSONDecodeError:
                pass

        # Retry with explicit JSON request
        if max_retries > 0:
            logger.warning("[FactExtractor] Non-dict response, retrying with JSON prompt...")
            retry_prompt = RETRY_JSON_PROMPT.format(
                required_format=required_format, original_prompt=original_prompt[:500]  # Truncate to save tokens
            )
            try:
                retry_result = await self.llm_service.ainvoke(
                    retry_prompt, response_format="json", priority=LLMPriority.LOW
                )
                return await self._parse_llm_response(retry_result, original_prompt, required_format, max_retries - 1)
            except Exception as e:
                logger.warning(f"[FactExtractor] Retry failed: {e}")

        return None

    async def extract(self, scraped_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract facts from scraped web pages using BATCHED LLM call.
        ONE LLM call for ALL pages (instead of N calls).

        Args:
            scraped_pages: List of dicts with 'content', 'source', 'url', 'published_at'

        Returns:
            List of extracted facts with statement, confidence, source info
        """
        if not scraped_pages:
            return []

        # Filter valid pages and prepare for batching
        valid_pages = []
        for page in scraped_pages:
            content = page.get("content", "")
            if content and len(content) >= 10:
                valid_pages.append(page)

        if not valid_pages:
            return []

        logger.info(f"[FactExtractor] Batch extracting facts from {len(valid_pages)} pages (1 LLM call)")

        # Build batch prompt with all page contents
        sections = []
        for i, page in enumerate(valid_pages):
            content = page.get("content", "")
            content_chunk = truncate_content(content, max_length=1500)  # Smaller chunks for batching
            sections.append(f"[{i}] {content_chunk}")

        batch_content = "\n\n".join(sections)
        prompt = f"{BATCH_FACT_PROMPT}{batch_content}"

        required_format = '{"results": [{"index": 0, "facts": [{"statement": "...", "confidence": 0.85}]}]}'

        try:
            # Single batched LLM call for ALL pages
            result = await self.llm_service.ainvoke(prompt, response_format="json", priority=LLMPriority.LOW)

            # Parse with retry logic
            parsed = await self._parse_llm_response(result, prompt, required_format)

            if parsed is None:
                logger.warning("[FactExtractor] Could not parse LLM response after retries, using fallback")
                return await self._extract_per_page_fallback(valid_pages)

            facts: List[Dict[str, Any]] = []
            results_list = parsed.get("results", [])

            for item in results_list:
                idx = item.get("index", -1)
                extracted = item.get("facts", [])

                if idx < 0 or idx >= len(valid_pages):
                    continue

                page = valid_pages[idx]

                for fact in extracted:
                    if not isinstance(fact, dict):
                        continue
                    fact["statement"] = clean_statement(fact.get("statement", ""))
                    fact["source_url"] = page.get("url", "")
                    fact["source"] = page.get("source", "")
                    fact["published_at"] = page.get("published_at", "")
                    fact["fact_id"] = f"f_{len(facts)}"
                    facts.append(fact)

            logger.info(f"[FactExtractor] Extracted {len(facts)} facts from {len(valid_pages)} pages (batched)")
            return facts

        except (KeyError, json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[FactExtractor] Batch extraction failed: {e}, falling back to per-page")
            return await self._extract_per_page_fallback(valid_pages)

    async def _extract_per_page_fallback(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback: extract facts one page at a time if batch fails."""
        facts: List[Dict[str, Any]] = []

        for page in pages:
            content = page.get("content", "")
            content_chunk = truncate_content(content, max_length=2000)
            prompt = FACT_EXTRACTION_PROMPT.format(content=content_chunk)

            try:
                result = await self.llm_service.ainvoke(prompt, response_format="json", priority=LLMPriority.LOW)
                extracted = result.get("facts", [])

                for fact in extracted:
                    if not isinstance(fact, dict):
                        continue
                    fact["statement"] = clean_statement(fact.get("statement", ""))
                    fact["source_url"] = page.get("url", "")
                    fact["source"] = page.get("source", "")
                    fact["published_at"] = page.get("published_at", "")
                    fact["fact_id"] = f"f_{len(facts)}"
                    facts.append(fact)
            except Exception as e:
                logger.warning(f"[FactExtractor] Failed to extract from {page.get('url')}: {e}")
                continue

        logger.info(f"[FactExtractor] Extracted {len(facts)} facts (fallback mode)")
        return facts
