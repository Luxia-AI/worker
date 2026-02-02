from typing import Any, Dict, List

from app.constants.llm_prompts import BIOMED_NER_PROMPT
from app.core.logger import get_logger
from app.services.common.list_ops import dedupe_list
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)


class EntityExtractor:
    """
    Biomedical entity extraction powered by Groq LLM.
    More portable than SciSpaCy and works on all platforms.
    Uses HIGH priority (Groq) because it's in the critical request path.
    """

    def __init__(self) -> None:
        self.llm = HybridLLMService()

    async def extract_entities(self, statement: str) -> List[str]:
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
        logger.info(f"[EntityExtractor] Annotating entities for {len(facts)} facts")

        updated = []
        for fact in facts:
            statement = fact.get("statement", "")
            fact["entities"] = await self.extract_entities(statement)
            updated.append(fact)

        return updated
