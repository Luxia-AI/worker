import json
import uuid
from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.common.dedup import dedup_triples_by_structure
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)

# Batch relation extraction prompt - extract triples from multiple facts at once
BATCH_TRIPLE_PROMPT = """You are a relation extraction agent specialized in biomedical/health facts.
Extract entity-relation-entity triples from each fact below.

Requirements:
- Subject/object must be entity strings
- Relation should be concise (e.g., "causes", "reduces risk of", "is treatment for")
- Confidence: float 0-1 indicating support strength

Return ONLY valid JSON (no markdown):
{"results": [{"index": 0, "triples": [{"subject": "...", "relation": "...", "object": "..."}]}]}

ENTITIES PROVIDED: {entities}

FACTS:
"""


class RelationExtractor:
    """
    LLM-based relation extractor that converts atomic facts into KG-ready triples.
    Uses BATCH processing to minimize LLM calls and save Groq quota.

    Public methods:
      - extract_relations(facts, entities) -> List[triples]
    """

    def __init__(self) -> None:
        self.llm_service = HybridLLMService()

    async def extract_relations(self, facts: List[Dict[str, Any]], entities: List[str]) -> List[Dict[str, Any]]:
        """
        Extract triples for ALL facts in ONE batched LLM call.

        Args:
            facts: list of fact dicts (must contain 'statement' and optionally 'fact_id', 'source_url')
            entities: global list of entities to provide context (strings)

        Returns:
            List of triples with subject, relation, object, confidence, source info
        """
        if not facts:
            return []

        logger.info(f"[RelationExtractor] Batch extracting relations for {len(facts)} facts (1 LLM call)")

        # Build batch prompt with all facts
        facts_text = "\n".join(f"[{i}] {fact.get('statement', '')}" for i, fact in enumerate(facts))
        entities_str = ", ".join(entities[:50])  # Limit entities to avoid token overflow
        prompt = BATCH_TRIPLE_PROMPT.format(entities=entities_str) + facts_text

        try:
            # Single batched LLM call for ALL facts
            result = await self.llm_service.ainvoke(prompt, response_format="json", priority=LLMPriority.LOW)

            # Parse batch results - handle case where LLM returns string or malformed response
            all_triples: List[Dict[str, Any]] = []

            if not isinstance(result, dict):
                logger.warning(f"[RelationExtractor] LLM returned non-dict: {type(result)}")
                return []

            results_list = result.get("results", [])

            for item in results_list:
                idx = item.get("index", -1)
                raw_triples = item.get("triples", [])

                if idx < 0 or idx >= len(facts):
                    continue

                fact = facts[idx]
                src = fact.get("source_url") or fact.get("source") or None

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
                        all_triples.append(triple)
                    except Exception as e:
                        logger.warning(f"[RelationExtractor] Skipping malformed triple: {e}")
                        continue

            deduped = dedup_triples_by_structure(all_triples)
            logger.info(f"[RelationExtractor] Extracted {len(deduped)} unique triples (batched)")
            return deduped

        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[RelationExtractor] Batch extraction failed: {e}")
            return []
        except Exception as e:
            logger.error(f"[RelationExtractor] LLM call failed: {e}")
            return []
