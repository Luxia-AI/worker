import json
import os
import re
from typing import Any, Dict, List, Optional

from app.constants.llm_prompts import FACT_EXTRACTION_PROMPT
from app.core.config import settings
from app.core.logger import get_logger
from app.services.common.dedup import generate_fact_id
from app.services.common.text_cleaner import clean_statement, truncate_content
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)

# Batch fact extraction prompt - extract facts from multiple pages at once
BATCH_FACT_PROMPT = """Extract key factual statements from each content section below.
For each section, return ONLY atomic, single-claim facts (no conjunctions).

IMPORTANT: You MUST respond with valid JSON only. No markdown, no explanations.
Return ONLY valid JSON with this exact structure:
{{"results": [{{"index": 0, "facts": [{{"statement": "...", "confidence": 0.85}}]}}, {{"index": 1, "facts": [...]}}]}}

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

    def _try_parse_result(self, result: Any) -> Optional[Dict[str, Any]]:
        """Try to parse LLM result into expected format. No retries, just parsing."""
        # Reject None, empty dict, or error indicators
        if result is None or result == {} or (isinstance(result, dict) and "_llm_error" in result):
            return None

        # If already a valid dict with results, return it
        if isinstance(result, dict) and "results" in result:
            return result

        # If dict but missing results key, try to extract
        if isinstance(result, dict):
            # Maybe LLM returned {"facts": [...]} for single page
            if "facts" in result:
                return {"results": [{"index": 0, "facts": result["facts"]}]}
            logger.warning(f"[FactExtractor] Dict missing 'results' key: {list(result.keys())}")
            return None

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

        return None

    @staticmethod
    def _confidence_mode() -> bool:
        value = os.getenv("LUXIA_CONFIDENCE_MODE")
        if value is None:
            return bool(getattr(settings, "LUXIA_CONFIDENCE_MODE", False))
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _chunk_content(content: str, max_chars: int, max_chunks: int) -> List[str]:
        text = (content or "").strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]
        chunks: List[str] = []
        start = 0
        overlap = min(240, max_chars // 6)
        while start < len(text) and len(chunks) < max_chunks:
            end = min(len(text), start + max_chars)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(0, end - overlap)
        return chunks or [text[:max_chars]]

    def _split_atomic_statement(self, statement: str) -> List[str]:
        text = (statement or "").strip()
        if not text:
            return []

        # Split on semicolons first
        parts = [p.strip() for p in text.split(";") if p.strip()]
        if len(parts) > 1:
            return parts

        # Split on " and " only when there are multiple numbers present
        num_count = len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text))
        if num_count >= 2 and " and " in text.lower():
            segments = [p.strip() for p in re.split(r"\band\b", text, flags=re.IGNORECASE) if p.strip()]
            # Avoid splitting common paired-body phrases
            if any("hands and feet" in text.lower() for _ in [0]):
                return [text]
            return segments

        return [text]

    def _normalize_atomic_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        low_signal_phrases = (
            "data element definitions",
            "registration or results information",
            "javascript and cookies",
            "requires human verification",
            "enable javascript",
        )
        normalized: List[Dict[str, Any]] = []
        for fact in facts:
            stmt = clean_statement(fact.get("statement", ""))
            for part in self._split_atomic_statement(stmt):
                if len(part) < 10:
                    continue
                if any(p in part.lower() for p in low_signal_phrases):
                    continue
                new_fact = dict(fact)
                new_fact["statement"] = clean_statement(part)
                new_fact["fact_id"] = generate_fact_id(new_fact["statement"], new_fact.get("source_url", ""))
                normalized.append(new_fact)
        return normalized

    async def _parse_llm_response(
        self, result: Any, original_prompt: str, required_format: str, max_retries: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Parse LLM response with limited retries. NO RECURSION - uses iterative loop."""
        # First attempt: try to parse the original result
        parsed = self._try_parse_result(result)
        if parsed is not None:
            return parsed

        # Retry loop (no recursion)
        for attempt in range(max_retries):
            logger.warning(f"[FactExtractor] Retry {attempt + 1}/{max_retries}: asking LLM for valid JSON...")
            retry_prompt = RETRY_JSON_PROMPT.format(
                required_format=required_format, original_prompt=original_prompt[:500]
            )
            try:
                retry_result = await self.llm_service.ainvoke(
                    retry_prompt,
                    response_format="json",
                    priority=LLMPriority.LOW,
                    call_tag="fact_extraction",
                )
                parsed = self._try_parse_result(retry_result)
                if parsed is not None:
                    logger.info(f"[FactExtractor] Retry {attempt + 1} succeeded")
                    return parsed
            except Exception as e:
                logger.warning(f"[FactExtractor] Retry {attempt + 1} failed: {e}")

        logger.warning("[FactExtractor] All retries exhausted, returning None")
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

        confidence_mode = self._confidence_mode()
        chunk_chars = int(os.getenv("CONFIDENCE_FACT_CHUNK_CHARS", "2600")) if confidence_mode else 1500
        max_chunks_per_page = int(os.getenv("CONFIDENCE_MAX_CHUNKS_PER_PAGE", "3")) if confidence_mode else 1
        target_facts_per_page = int(os.getenv("CONFIDENCE_FACTS_PER_PAGE_TARGET", "20")) if confidence_mode else 8

        logger.info(
            "[FactExtractor] Batch extracting facts from %d pages (1 LLM call, confidence_mode=%s, "
            "chunk_chars=%d, max_chunks_per_page=%d, target_facts_per_page=%d)",
            len(valid_pages),
            confidence_mode,
            chunk_chars,
            max_chunks_per_page,
            target_facts_per_page,
        )

        # Build batch prompt with all page contents
        sections = []
        section_to_page_idx: List[int] = []
        for page_idx, page in enumerate(valid_pages):
            content = page.get("content", "")
            base_content = truncate_content(content, max_length=chunk_chars * max_chunks_per_page + 200)
            chunks = self._chunk_content(base_content, max_chars=chunk_chars, max_chunks=max_chunks_per_page)
            for chunk in chunks:
                section_index = len(section_to_page_idx)
                section_to_page_idx.append(page_idx)
                sections.append(f"[{section_index}] {chunk}")

        batch_content = "\n\n".join(sections)
        prompt = (
            f"{BATCH_FACT_PROMPT}{batch_content}\n\n"
            f"TARGET_FACTS_PER_SECTION: up to {target_facts_per_page} high-quality atomic facts."
        )

        required_format = '{"results": [{"index": 0, "facts": [{"statement": "...", "confidence": 0.85}]}]}'

        try:
            # Single batched LLM call for ALL pages
            result = await self.llm_service.ainvoke(
                prompt,
                response_format="json",
                priority=LLMPriority.LOW,
                call_tag="fact_extraction",
            )

            # Parse with retry logic
            parsed = await self._parse_llm_response(result, prompt, required_format)

            if parsed is None and confidence_mode:
                logger.warning(
                    "[FactExtractor] Confidence mode retry: overriding Groq reservation to avoid zero-fact extraction"
                )
                retry_result = await self.llm_service.ainvoke(
                    prompt,
                    response_format="json",
                    priority=LLMPriority.HIGH,
                    call_tag="fact_extraction",
                    allow_quota_override=True,
                )
                parsed = await self._parse_llm_response(retry_result, prompt, required_format)

            if parsed is None:
                logger.warning("[FactExtractor] Could not parse LLM response after retries, using fallback")
                return await self._extract_per_page_fallback(valid_pages)

            facts: List[Dict[str, Any]] = []
            results_list = parsed.get("results", [])

            for item in results_list:
                idx = item.get("index", -1)
                extracted = item.get("facts", [])

                if idx < 0 or idx >= len(section_to_page_idx):
                    continue

                page = valid_pages[section_to_page_idx[idx]]

                for fact in extracted:
                    if not isinstance(fact, dict):
                        continue
                    fact["statement"] = clean_statement(fact.get("statement", ""))
                    fact["source_url"] = page.get("url", "")
                    fact["source"] = page.get("source", "")
                    fact["published_at"] = page.get("published_at", "")
                    # Use deterministic fact_id based on content hash (prevents duplicates)
                    fact["fact_id"] = generate_fact_id(fact["statement"], fact["source_url"])
                    facts.append(fact)

            facts = self._normalize_atomic_facts(facts)
            dedup_by_id: Dict[str, Dict[str, Any]] = {}
            for fact in facts:
                fact_id = str(fact.get("fact_id", ""))
                if fact_id and fact_id not in dedup_by_id:
                    dedup_by_id[fact_id] = fact
            if dedup_by_id:
                facts = list(dedup_by_id.values())
            logger.info(f"[FactExtractor] Extracted {len(facts)} facts from {len(valid_pages)} pages (batched)")
            return facts

        except Exception as e:
            logger.warning(f"[FactExtractor] Batch extraction failed: {e}, falling back to per-page")
            return await self._extract_per_page_fallback(valid_pages)

    async def _extract_per_page_fallback(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback: extract facts one page at a time if batch fails."""
        facts: List[Dict[str, Any]] = []
        confidence_mode = self._confidence_mode()
        allow_override = confidence_mode

        for page in pages:
            content = page.get("content", "")
            content_chunk = truncate_content(content, max_length=2000)
            prompt = FACT_EXTRACTION_PROMPT.format(content=content_chunk)

            try:
                result = await self.llm_service.ainvoke(
                    prompt,
                    response_format="json",
                    priority=LLMPriority.HIGH if allow_override else LLMPriority.LOW,
                    call_tag="fact_extraction",
                    allow_quota_override=allow_override,
                )
                extracted = result.get("facts", [])

                for fact in extracted:
                    if not isinstance(fact, dict):
                        continue
                    fact["statement"] = clean_statement(fact.get("statement", ""))
                    fact["source_url"] = page.get("url", "")
                    fact["source"] = page.get("source", "")
                    fact["published_at"] = page.get("published_at", "")
                    # Use deterministic fact_id based on content hash (prevents duplicates)
                    fact["fact_id"] = generate_fact_id(fact["statement"], fact["source_url"])
                    facts.append(fact)
                if extracted:
                    allow_override = False
            except Exception as e:
                logger.warning(f"[FactExtractor] Failed to extract from {page.get('url')}: {e}")
                continue

        facts = self._normalize_atomic_facts(facts)
        logger.info(f"[FactExtractor] Extracted {len(facts)} facts (fallback mode)")
        return facts
