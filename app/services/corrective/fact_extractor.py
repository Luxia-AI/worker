import json
import os
import re
from typing import Any, Dict, List, Optional

from app.constants.llm_prompts import FACT_EXTRACTION_PREDICATE_FORCING_PROMPT, FACT_EXTRACTION_PROMPT
from app.core.config import settings
from app.core.logger import get_logger, log_value_payload
from app.services.common.dedup import generate_fact_id
from app.services.common.text_cleaner import clean_statement, truncate_content
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)

# Batch fact extraction prompt - extract facts from multiple pages at once
BATCH_FACT_PROMPT = """Extract key factual statements from each content section below.
For each section, return ONLY atomic, single-claim facts (no conjunctions).
Keep only truth-grounded statements explicitly asserted in the section.
Exclude speculative/hedged language (may, might, could, possible, suggests, appears),
opinion text, claim-about-claim text, and generic background filler.

Claim alignment rules (strict):
- Keep statements directly useful for verifying the claim context.
- Prefer statements containing claim intervention/outcome entities or close lexical equivalents.
- Drop off-topic biomedical facts even if true in isolation.
- Keep explicit support and explicit contradiction statements when claim-relevant.

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

    @staticmethod
    def _is_truth_grounded_statement(statement: str) -> bool:
        text = (statement or "").strip()
        if len(text) < 10:
            return False
        lower = text.lower()
        if "?" in lower:
            return False
        hedge_patterns = (
            r"\bmay\b",
            r"\bmight\b",
            r"\bcould\b",
            r"\bpossible\b",
            r"\bpossibly\b",
            r"\bpotentially\b",
            r"\blikely\b",
            r"\bunlikely\b",
            r"\bsuggest(?:s|ed|ing)?\b",
            r"\bappear(?:s|ed|ing)?\b",
            r"\bindicate(?:s|d)?\b",
            r"\bhypothes(?:is|ized|es)\b",
        )
        if any(re.search(p, lower) for p in hedge_patterns):
            return False
        non_factual_patterns = (
            r"\b(some people|many people)\s+(say|claim|believe)\b",
            r"\bit is believed\b",
            r"\brumou?r\b",
            r"\bconspiracy\b",
            r"\bopinion\b",
        )
        if any(re.search(p, lower) for p in non_factual_patterns):
            return False
        return True

    def _filter_truth_grounded_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for fact in facts:
            stmt = str(fact.get("statement", "") or "")
            if self._is_truth_grounded_statement(stmt):
                kept.append(fact)
        return kept

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        stop = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "to",
            "for",
            "of",
            "in",
            "on",
            "with",
            "by",
            "at",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }
        return {w for w in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", (text or "").lower()) if w not in stop}

    @staticmethod
    def _predicate_family_tokens(text: str) -> set[str]:
        low = (text or "").lower()
        families: set[str] = set()
        if re.search(r"\b(reduc|lower|decreas|alleviat|improv|help|benefit|reliev)\w*\b", low):
            families.add("improve_reduce")
        if re.search(r"\b(increas|worsen|exacerbat|aggravat|trigger|cause)\w*\b", low):
            families.add("increase_cause")
        if re.search(r"\b(treat|cure|prevent|protect|work|effective|effic)\w*\b", low):
            families.add("treat_prevent")
        if re.search(r"\b(associat|link|correlat|relat)\w*\b", low):
            families.add("associate")
        return families

    def _filter_claim_admissible_facts(
        self,
        facts: List[Dict[str, Any]],
        claim_text: str = "",
        claim_entities: Optional[List[str]] = None,
        must_have_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        claim_tokens = self._tokenize(claim_text)
        claim_predicates = self._predicate_family_tokens(claim_text)
        claim_entity_tokens: set[str] = set()
        for ent in claim_entities or []:
            claim_entity_tokens |= self._tokenize(ent)
        must_have_tokens: set[str] = set()
        for ent in must_have_entities or []:
            must_have_tokens |= self._tokenize(ent)

        kept: List[Dict[str, Any]] = []
        for fact in facts:
            stmt = str(fact.get("statement", "") or "")
            stmt_tokens = self._tokenize(stmt)
            if not stmt_tokens:
                continue

            has_must_have = bool(must_have_tokens and (stmt_tokens & must_have_tokens))
            has_claim_entity = bool(claim_entity_tokens and (stmt_tokens & claim_entity_tokens))

            # Primary-entity gate: must-have if available, otherwise any claim entity.
            if must_have_tokens:
                if not has_must_have:
                    continue
            elif claim_entity_tokens and not has_claim_entity:
                continue

            if claim_tokens and len(stmt_tokens & claim_tokens) == 0:
                continue

            stmt_predicates = self._predicate_family_tokens(stmt)
            if claim_predicates and not (stmt_predicates & claim_predicates):
                # Relax only when anchor/entity alignment is strong to avoid zero-fact runs.
                lexical_overlap = len(stmt_tokens & claim_tokens)
                if not ((has_must_have or has_claim_entity) and lexical_overlap >= 2):
                    continue
            kept.append(fact)
        # Failsafe: preserve anchor-matching facts when strict predicate filtering empties output.
        if kept:
            return kept
        if not facts:
            return []

        fallback: List[Dict[str, Any]] = []
        for fact in facts:
            stmt = str(fact.get("statement", "") or "")
            stmt_tokens = self._tokenize(stmt)
            if not stmt_tokens:
                continue
            if must_have_tokens and not (stmt_tokens & must_have_tokens):
                continue
            if claim_tokens and len(stmt_tokens & claim_tokens) < 2:
                continue
            fallback.append(fact)
        return fallback

    @staticmethod
    def _build_claim_context_block(
        claim_text: str = "",
        claim_entities: Optional[List[str]] = None,
        must_have_entities: Optional[List[str]] = None,
    ) -> str:
        claim = str(claim_text or "").strip()
        claim_entities = [str(e).strip() for e in (claim_entities or []) if str(e).strip()]
        must_have_entities = [str(e).strip() for e in (must_have_entities or []) if str(e).strip()]
        if not claim and not claim_entities and not must_have_entities:
            return ""
        return (
            "\n\nCLAIM CONTEXT (must guide extraction):\n"
            f"- claim: {claim}\n"
            f"- claim_entities: {', '.join(claim_entities[:10]) if claim_entities else '[]'}\n"
            f"- must_have_entities: {', '.join(must_have_entities[:10]) if must_have_entities else '[]'}\n"
            "- Keep only statements that help verify or refute this claim.\n"
            "- Include contradictory evidence if it is claim-relevant.\n"
        )

    @classmethod
    def _build_fact_prompt(
        cls,
        content: str,
        predicate_target: Optional[Dict[str, str]] = None,
        claim_text: str = "",
        claim_entities: Optional[List[str]] = None,
        must_have_entities: Optional[List[str]] = None,
    ) -> str:
        claim_block = cls._build_claim_context_block(
            claim_text=claim_text,
            claim_entities=claim_entities,
            must_have_entities=must_have_entities,
        )
        if predicate_target:
            return (
                FACT_EXTRACTION_PREDICATE_FORCING_PROMPT.format(
                    subject=str(predicate_target.get("subject", "") or ""),
                    predicate=str(predicate_target.get("predicate", "") or ""),
                    object=str(predicate_target.get("object", "") or ""),
                    content=content,
                )
                + claim_block
            )
        return FACT_EXTRACTION_PROMPT.format(content=content) + claim_block

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

    async def extract(
        self,
        scraped_pages: List[Dict[str, Any]],
        predicate_target: Optional[Dict[str, str]] = None,
        claim_text: str = "",
        claim_entities: Optional[List[str]] = None,
        must_have_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
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

        log_value_payload(
            logger,
            "fact_extraction",
            {
                "pages_input": len(valid_pages),
                "confidence_mode": confidence_mode,
                "chunk_chars": chunk_chars,
                "max_chunks_per_page": max_chunks_per_page,
                "target_facts_per_page": target_facts_per_page,
                "predicate_forcing_mode": bool(predicate_target),
                "predicate_target": predicate_target or {},
            },
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
        claim_context_block = self._build_claim_context_block(
            claim_text=claim_text,
            claim_entities=claim_entities,
            must_have_entities=must_have_entities,
        )
        prompt = (
            f"{BATCH_FACT_PROMPT}{batch_content}\n\n"
            f"TARGET_FACTS_PER_SECTION: up to {target_facts_per_page} high-quality atomic facts."
            f"{claim_context_block}"
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
                return await self._extract_per_page_fallback(
                    valid_pages,
                    predicate_target=predicate_target,
                    claim_text=claim_text,
                    claim_entities=claim_entities,
                    must_have_entities=must_have_entities,
                )

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
            facts = self._filter_truth_grounded_facts(facts)
            facts = self._filter_claim_admissible_facts(
                facts,
                claim_text=claim_text,
                claim_entities=claim_entities,
                must_have_entities=must_have_entities,
            )
            dedup_by_id: Dict[str, Dict[str, Any]] = {}
            for fact in facts:
                fact_id = str(fact.get("fact_id", ""))
                if fact_id and fact_id not in dedup_by_id:
                    dedup_by_id[fact_id] = fact
            if dedup_by_id:
                facts = list(dedup_by_id.values())
            confidences = [float(f.get("confidence") or 0.0) for f in facts if f.get("confidence") is not None]
            log_value_payload(
                logger,
                "fact_extraction",
                {
                    "facts_count": len(facts),
                    "pages_input": len(valid_pages),
                    "facts_sample": facts,
                    "fact_conf_min": min(confidences) if confidences else 0.0,
                    "fact_conf_max": max(confidences) if confidences else 0.0,
                    "fact_conf_avg": (sum(confidences) / len(confidences)) if confidences else 0.0,
                },
            )
            return facts

        except Exception as e:
            logger.warning(f"[FactExtractor] Batch extraction failed: {e}, falling back to per-page")
            return await self._extract_per_page_fallback(
                valid_pages,
                predicate_target=predicate_target,
                claim_text=claim_text,
                claim_entities=claim_entities,
                must_have_entities=must_have_entities,
            )

    async def _extract_per_page_fallback(
        self,
        pages: List[Dict[str, Any]],
        predicate_target: Optional[Dict[str, str]] = None,
        claim_text: str = "",
        claim_entities: Optional[List[str]] = None,
        must_have_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback: extract facts one page at a time if batch fails."""
        facts: List[Dict[str, Any]] = []
        confidence_mode = self._confidence_mode()
        allow_override = confidence_mode

        for page in pages:
            content = page.get("content", "")
            content_chunk = truncate_content(content, max_length=2000)
            prompt = self._build_fact_prompt(
                content_chunk,
                predicate_target=predicate_target,
                claim_text=claim_text,
                claim_entities=claim_entities,
                must_have_entities=must_have_entities,
            )

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
        facts = self._filter_truth_grounded_facts(facts)
        facts = self._filter_claim_admissible_facts(
            facts,
            claim_text=claim_text,
            claim_entities=claim_entities,
            must_have_entities=must_have_entities,
        )
        log_value_payload(
            logger,
            "fact_extraction_fallback",
            {
                "facts_count": len(facts),
                "facts_sample": facts,
                "predicate_forcing_mode": bool(predicate_target),
                "predicate_target": predicate_target or {},
            },
        )
        return facts
