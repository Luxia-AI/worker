from typing import Any, Dict, List

from app.constants.llm_prompts import BIOMED_NER_PROMPT
from app.core.logger import get_logger
from app.services.corrective.fact_extractor import FactExtractor

logger = get_logger(__name__)


class EntityExtractor:
    """
    Biomedical entity extraction powered by Groq LLM.
    More portable than SciSpaCy and works on all platforms.
    """

    def __init__(self) -> None:
        self.llm = FactExtractor()

    async def extract_entities(self, statement: str) -> List[str]:
        prompt = f"{BIOMED_NER_PROMPT}\n\nFACT:\n{statement}"

        try:
            result = await self.llm.ainvoke(prompt, response_format="json")
            ents = result.get("entities", [])
            cleaned = [e.lower().strip() for e in ents if isinstance(e, str)]
            return list(dict.fromkeys(cleaned))  # dedupe
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
