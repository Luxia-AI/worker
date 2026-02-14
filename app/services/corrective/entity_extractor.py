import json
from typing import Any, Dict, List, Optional

from app.constants.llm_prompts import BIOMED_NER_PROMPT
from app.core.logger import get_logger
from app.services.common.list_ops import dedupe_list
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)

# Batch entity extraction prompt - extract entities for multiple facts at once
BATCH_NER_PROMPT = """Extract biomedical entities from each fact below.
For each fact, identify: genes, proteins, diseases, drugs, chemicals, biological processes, anatomical terms.
Keep only entities explicitly present in factual assertions.
Do not infer entities from speculative/hedged clauses.
Do not return generic meta terms (e.g., study, research, evidence, article, claim).

IMPORTANT: You MUST respond with valid JSON only. No markdown, no explanations.
Respond with JSON in this exact format:
{{"results": [{{"index": 0, "entities": ["entity1", "entity2"]}}, {{"index": 1, "entities": [...]}}]}}

FACTS:
"""

# Retry prompt when LLM returns non-dict
RETRY_JSON_PROMPT = """Your previous response was not valid JSON. Please respond ONLY with valid JSON.
No markdown code blocks, no explanations, just the JSON object.

Required format:
{required_format}

Original request:
{original_prompt}
"""


class EntityExtractor:
    """
    Biomedical entity extraction powered by Groq LLM.
    More portable than SciSpaCy and works on all platforms.
    Uses HIGH priority (Groq) because it's in the critical request path.
    Uses BATCH processing to minimize LLM calls.
    """

    def __init__(self) -> None:
        self.llm = HybridLLMService()

    @staticmethod
    def _filter_truth_entities(entities: List[str]) -> List[str]:
        blocked = {
            "study",
            "studies",
            "research",
            "evidence",
            "article",
            "paper",
            "claim",
            "claims",
            "myth",
            "rumor",
            "rumour",
            "opinion",
        }
        cleaned: List[str] = []
        for entity in entities:
            e = str(entity).lower().strip()
            if not e or e in blocked:
                continue
            if len(e) < 2:
                continue
            cleaned.append(e)
        return dedupe_list(cleaned)

    async def extract_entities(self, statement: str) -> List[str]:
        """Extract entities from a single statement (used for claim extraction)."""
        prompt = f"{BIOMED_NER_PROMPT}\n\nFACT:\n{statement}"

        try:
            logger.info("[EntityExtractor] Calling LLM for entity extraction...")
            # HIGH priority: Entity extraction is in critical path, needs to be fast
            result = await self.llm.ainvoke(
                prompt,
                response_format="json",
                priority=LLMPriority.HIGH,
                call_tag="entity_extraction",
            )
            logger.info(f"[EntityExtractor] LLM returned: {result}")
            ents = result.get("entities", [])
            cleaned = [e.lower().strip() for e in ents if isinstance(e, str)]
            cleaned = self._filter_truth_entities(cleaned)
            logger.info(f"[EntityExtractor] Extracted entities: {cleaned}")
            return dedupe_list(cleaned)
        except Exception as e:
            logger.error(f"[EntityExtractor] Failed extraction: {e}")
            return []

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
            # Maybe LLM returned {"entities": [...]} for single fact
            if "entities" in result:
                return {"results": [{"index": 0, "entities": result["entities"]}]}
            logger.warning(f"[EntityExtractor] Dict missing 'results' key: {list(result.keys())}")
            return None

        # If string, try to parse as JSON
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    if "results" in parsed:
                        return parsed
                    if "entities" in parsed:
                        return {"results": [{"index": 0, "entities": parsed["entities"]}]}
            except json.JSONDecodeError:
                pass

        return None

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
            logger.warning(f"[EntityExtractor] Retry {attempt + 1}/{max_retries}: asking LLM for valid JSON...")
            retry_prompt = RETRY_JSON_PROMPT.format(
                required_format=required_format, original_prompt=original_prompt[:500]
            )
            try:
                retry_result = await self.llm.ainvoke(
                    retry_prompt,
                    response_format="json",
                    priority=LLMPriority.HIGH,
                    call_tag="entity_extraction",
                )
                parsed = self._try_parse_result(retry_result)
                if parsed is not None:
                    logger.info(f"[EntityExtractor] Retry {attempt + 1} succeeded")
                    return parsed
            except Exception as e:
                logger.warning(f"[EntityExtractor] Retry {attempt + 1} failed: {e}")

        logger.warning("[EntityExtractor] All retries exhausted, returning None")
        return None

    async def annotate_entities(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Annotate entities for all facts in ONE batched LLM call (instead of N calls)."""
        if not facts:
            return []

        logger.info(f"[EntityExtractor] Batch annotating entities for {len(facts)} facts (1 LLM call)")

        # Build batch prompt with all facts
        facts_text = "\n".join(f"{i}. {fact.get('statement', '')}" for i, fact in enumerate(facts))
        prompt = f"{BATCH_NER_PROMPT}{facts_text}"
        required_format = '{"results": [{"index": 0, "entities": ["entity1", "entity2"]}]}'

        try:
            # Single LLM call for all facts
            result = await self.llm.ainvoke(
                prompt,
                response_format="json",
                priority=LLMPriority.HIGH,
                call_tag="entity_extraction",
            )

            # Parse with retry logic
            parsed = await self._parse_llm_response(result, prompt, required_format)

            if parsed is None:
                logger.warning("[EntityExtractor] Could not parse LLM response after retries")
                for fact in facts:
                    fact["entities"] = []
                return facts

            results_list = parsed.get("results", [])
            logger.info(f"[EntityExtractor] Batch LLM returned results for {len(results_list)} facts")

            # Map results back to facts
            results_map = {}
            for item in results_list:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index", -1)
                ents = item.get("entities", [])
                if not isinstance(ents, list):
                    ents = []
                cleaned = [e.lower().strip() for e in ents if isinstance(e, str)]
                results_map[idx] = self._filter_truth_entities(cleaned)

            # Apply entities to facts
            for i, fact in enumerate(facts):
                fact["entities"] = results_map.get(i, [])

            logger.info(f"[EntityExtractor] Batch extraction complete for {len(facts)} facts")
            return facts

        except Exception as e:
            logger.error(f"[EntityExtractor] Batch extraction failed: {e}")
            # Fallback: empty entities for all facts
            for fact in facts:
                fact["entities"] = []
            return facts
