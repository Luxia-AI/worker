import json
import uuid
from typing import Any, Dict, List, Optional

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

IMPORTANT: You MUST respond with valid JSON only. No markdown, no explanations.
Return ONLY valid JSON:
{"results": [{"index": 0, "triples": [{"subject": "...", "relation": "...", "object": "...", "confidence": 0.9}]}]}

ENTITIES PROVIDED: {entities}

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


class RelationExtractor:
    """
    LLM-based relation extractor that converts atomic facts into KG-ready triples.
    Uses BATCH processing to minimize LLM calls and save Groq quota.

    Public methods:
      - extract_relations(facts, entities) -> List[triples]
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
            # Maybe LLM returned {"triples": [...]} for single fact
            if "triples" in result:
                return {"results": [{"index": 0, "triples": result["triples"]}]}
            logger.warning(f"[RelationExtractor] Dict missing 'results' key: {list(result.keys())}")
            return None

        # If string, try to parse as JSON
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    if "results" in parsed:
                        return parsed
                    if "triples" in parsed:
                        return {"results": [{"index": 0, "triples": parsed["triples"]}]}
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
            logger.warning(f"[RelationExtractor] Retry {attempt + 1}/{max_retries}: asking LLM for valid JSON...")
            retry_prompt = RETRY_JSON_PROMPT.format(
                required_format=required_format, original_prompt=original_prompt[:500]
            )
            try:
                retry_result = await self.llm_service.ainvoke(
                    retry_prompt, response_format="json", priority=LLMPriority.LOW
                )
                parsed = self._try_parse_result(retry_result)
                if parsed is not None:
                    logger.info(f"[RelationExtractor] Retry {attempt + 1} succeeded")
                    return parsed
            except Exception as e:
                logger.warning(f"[RelationExtractor] Retry {attempt + 1} failed: {e}")

        logger.warning("[RelationExtractor] All retries exhausted, returning None")
        return None

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

        required_format = (
            '{"results": [{"index": 0, "triples": '
            '[{"subject": "...", "relation": "...", "object": "...", "confidence": 0.9}]}]}'
        )

        try:
            # Single batched LLM call for ALL facts
            result = await self.llm_service.ainvoke(prompt, response_format="json", priority=LLMPriority.LOW)

            # Parse with retry logic
            parsed = await self._parse_llm_response(result, prompt, required_format)

            if parsed is None:
                logger.warning("[RelationExtractor] Could not parse LLM response after retries")
                return []

            all_triples: List[Dict[str, Any]] = []
            results_list = parsed.get("results", [])

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
