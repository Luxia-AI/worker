import asyncio
import uuid
from typing import Any, Dict, Iterable, List

from app.constants.llm_prompts import TRIPLE_EXTRACTION_PROMPT
from app.core.logger import get_logger
from app.services.llms.groq_service import GroqService

logger = get_logger(__name__)


class RelationExtractor:
    """
    LLM-based relation extractor that converts atomic facts into KG-ready triples.

    Public methods:
      - extract_relations(facts, entities) -> List[triples]
    """

    def __init__(self, max_concurrent: int = 6):
        self.groq_service = GroqService()
        # concurrency guard when calling the LLM in parallel
        self._sem = asyncio.Semaphore(max_concurrent)

    async def _call_groq_service_for_fact(self, fact: Dict[str, Any], entities: List[str]) -> List[Dict[str, Any]]:
        """
        Call the LLM to extract triples for a single fact.
        """
        statement = fact.get("statement", "")
        src = fact.get("source_url") or fact.get("source") or None

        prompt = TRIPLE_EXTRACTION_PROMPT + f"\n\nSTATEMENT:\n{statement}\n\nENTITIES:\n{entities}\n"

        async with self._sem:
            try:
                # request JSON output from the model
                res = await self.groq_service.ainvoke(prompt, response_format="json")
            except Exception as e:
                logger.error(f"[RelationExtractor] LLM call failed for fact_id={fact.get('fact_id')}: {e}")
                return []

        # Validate & parse LLM response
        triples = []
        if not isinstance(res, dict):
            logger.error(
                f"[RelationExtractor] Unexpected LLM response type for fact_id={fact.get('fact_id')}: {type(res)}"
            )
            return []

        raw_triples = res.get("triples", [])
        if not isinstance(raw_triples, Iterable):
            logger.error(f"[RelationExtractor] LLM returned invalid 'triples' field for fact_id={fact.get('fact_id')}")
            return []

        for rt in raw_triples:
            try:
                subj = rt.get("subject", "").strip()
                rel = rt.get("relation", "").strip()
                obj = rt.get("object", "").strip()
                conf = float(rt.get("confidence", 0.0))
                if not subj or not rel or not obj:
                    continue

                triple = {
                    "id": str(uuid.uuid4()),
                    "subject": subj,
                    "relation": rel,
                    "object": obj,
                    "confidence": max(0.0, min(conf, 1.0)),
                    "source_url": src,
                    "fact_id": fact.get("fact_id"),
                }
                triples.append(triple)
            except Exception as e:
                logger.warning(
                    f"[RelationExtractor] Skipping malformed triple from LLM for fact_id={fact.get('fact_id')}: {e}"
                )
                continue

        return triples

    async def extract_relations(self, facts: List[Dict[str, Any]], entities: List[str]) -> List[Dict[str, Any]]:
        """
        Extract triples for a list of facts.

        Args:
            facts: list of fact dicts (must contain 'statement' and optionally 'fact_id', 'source_url')
            entities: global list of entities to provide context (strings)

        Returns:
            List of triples:
            {
              "id": "...",
              "subject": "...",
              "relation": "...",
              "object": "...",
              "confidence": 0.0-1.0,
              "source_url": "...",
              "fact_id": "..."
            }
        """
        logger.info(
            f"[RelationExtractor] Extracting relations for {len(facts)} facts (entities provided: {len(entities)})"
        )

        # If there are many facts, process them concurrently but respecting semaphore
        tasks = [self._call_groq_service_for_fact(fact, entities) for fact in facts]
        results = await asyncio.gather(*tasks)

        # Flatten results and deduplicate identical triples (subject, relation, object, source)
        all_triples = [t for sub in results for t in sub]
        unique: Dict[tuple[str, str, str, str | None], Dict[str, Any]] = {}
        for t in all_triples:
            key = (t["subject"].lower(), t["relation"].lower(), t["object"].lower(), t.get("source_url"))
            if key not in unique or unique[key]["confidence"] < t["confidence"]:
                unique[key] = t

        deduped = list(unique.values())
        logger.info(f"[RelationExtractor] Extracted {len(deduped)} unique triples")

        return deduped
