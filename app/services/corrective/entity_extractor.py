from typing import Any, Dict, List

from app.constants.llm_prompts import BIOMED_NER_PROMPT
from app.core.logger import get_logger
from app.services.common.list_ops import dedupe_list
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)

# Batch entity extraction prompt - extract entities for multiple facts at once
BATCH_NER_PROMPT = """Extract biomedical entities from each fact below.
For each fact, identify: genes, proteins, diseases, drugs, chemicals, biological processes, anatomical terms.

Respond with JSON in this exact format:
{"results": [{"index": 0, "entities": ["entity1", "entity2"]}, {"index": 1, "entities": [...]}]}

FACTS:
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

    async def extract_entities(self, statement: str) -> List[str]:
        """Extract entities from a single statement (used for claim extraction)."""
        prompt = f"{BIOMED_NER_PROMPT}\n\nFACT:\n{statement}"

        try:
            logger.info("[EntityExtractor] Calling LLM for entity extraction...")
            # HIGH priority: Entity extraction is in critical path, needs to be fast
            result = await self.llm.ainvoke(prompt, response_format="json", priority=LLMPriority.HIGH)
            logger.info(f"[EntityExtractor] LLM returned: {result}")
            ents = result.get("entities", [])
            cleaned = [e.lower().strip() for e in ents if isinstance(e, str)]
            logger.info(f"[EntityExtractor] Extracted entities: {cleaned}")
            return dedupe_list(cleaned)
        except Exception as e:
            logger.error(f"[EntityExtractor] Failed extraction: {e}")
            return []

    async def annotate_entities(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Annotate entities for all facts in ONE batched LLM call (instead of N calls)."""
        if not facts:
            return []

        logger.info(f"[EntityExtractor] Batch annotating entities for {len(facts)} facts (1 LLM call)")

        # Build batch prompt with all facts
        facts_text = "\n".join(f"{i}. {fact.get('statement', '')}" for i, fact in enumerate(facts))
        prompt = f"{BATCH_NER_PROMPT}{facts_text}"

        try:
            # Single LLM call for all facts
            result = await self.llm.ainvoke(prompt, response_format="json", priority=LLMPriority.HIGH)

            # Handle case where LLM returns string or malformed response
            if not isinstance(result, dict):
                logger.warning(f"[EntityExtractor] LLM returned non-dict: {type(result)}")
                for fact in facts:
                    fact["entities"] = []
                return facts

            logger.info(f"[EntityExtractor] Batch LLM returned results for {len(result.get('results', []))} facts")

            # Map results back to facts
            results_map = {}
            for item in result.get("results", []):
                idx = item.get("index", -1)
                ents = item.get("entities", [])
                cleaned = [e.lower().strip() for e in ents if isinstance(e, str)]
                results_map[idx] = dedupe_list(cleaned)

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
