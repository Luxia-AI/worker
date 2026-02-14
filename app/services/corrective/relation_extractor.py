import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger, log_value_payload
from app.services.common.dedup import dedup_triples_by_structure
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)
RELATION_MIN_CONF = float(os.getenv("RELATION_MIN_CONFIDENCE", "0.2"))

# Batch relation extraction prompt - extract triples from multiple facts at once
BATCH_TRIPLE_PROMPT = """You are a relation extraction agent specialized in biomedical/health facts.
Extract entity-relation-entity triples from each fact below.

Requirements:
- Subject/object must be entity strings
- Relation should be concise (e.g., "causes", "reduces risk of", "is treatment for")
- Confidence: float 0-1 indicating support strength

IMPORTANT: You MUST respond with valid JSON only. No markdown, no explanations.
Return ONLY valid JSON:
{{"results": [{{"index": 0, "triples": [{{"subject": "...", "relation": "...", "object": "...",
"confidence": 0.9}}]}}]}}

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

    @staticmethod
    def _normalize_relation(relation: str) -> str:
        return re.sub(r"\s+", " ", (relation or "").replace("_", " ").strip().lower())

    def _is_causal_relation(self, relation: str) -> bool:
        rel = self._normalize_relation(relation)
        causal_markers = {
            "cause",
            "causes",
            "caused",
            "causing",
            "contribute",
            "contributes",
            "contributed",
            "contributing",
            "induce",
            "induces",
            "induced",
            "trigger",
            "triggers",
            "triggered",
            "result in",
            "results in",
            "resulted in",
            "lead to",
            "leads to",
            "led to",
            "associate",
            "associated",
            "association",
            "linked",
            "link",
            "increase risk",
            "increases risk",
        }
        return any(marker in rel for marker in causal_markers)

    def _is_negated_statement(self, statement: str) -> bool:
        text = (statement or "").lower()
        negation_patterns = [
            r"\b(?:do|does|did|is|are|was|were|can|could|may|might|must|should|would|will)\s+not\b",
            r"\bno\s+(?:evidence|link|association|causal link)\b",
            r"\bnot\s+(?:associated|linked|causal)\b",
            r"\bunrelated to\b",
            r"\b(?:myth|debunked|false claim|no link)\b",
        ]
        return any(re.search(pat, text) for pat in negation_patterns)

    def _negated_relation_label(self, relation: str) -> Optional[str]:
        rel = self._normalize_relation(relation)
        if "cause" in rel:
            return "does_not_cause"
        if "contribut" in rel:
            return "does_not_cause"
        if "lead to" in rel:
            return "does_not_cause"
        if "result in" in rel:
            return "does_not_cause"
        if "associat" in rel:
            return "not_associated_with"
        if "link" in rel:
            return "not_linked_to"
        if "increase risk" in rel:
            return "does_not_increase_risk_of"
        return None

    def _apply_negation_guard(
        self,
        statement: str,
        relation: str,
    ) -> Optional[str]:
        """
        Prevent positive causal triples from negated fact statements.
        Returns normalized/inverted relation or None when triple should be dropped.
        """
        if not self._is_causal_relation(relation):
            return relation
        if not self._is_negated_statement(statement):
            return relation
        negated = self._negated_relation_label(relation)
        if negated:
            return negated
        return None

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
            except json.JSONDecodeError as e:
                preview = result[:800].replace("\n", "\\n")
                logger.warning(
                    "[RelationExtractor] JSON parse error: msg=%s line=%s col=%s pos=%s preview=%s",
                    e.msg,
                    e.lineno,
                    e.colno,
                    e.pos,
                    preview,
                )
                repaired = self._repair_json_string(result)
                if repaired:
                    try:
                        parsed = json.loads(repaired)
                        if isinstance(parsed, dict):
                            if "results" in parsed:
                                logger.warning("[RelationExtractor] JSON repaired successfully")
                                return parsed
                            if "triples" in parsed:
                                logger.warning("[RelationExtractor] JSON repaired successfully (triples root)")
                                return {"results": [{"index": 0, "triples": parsed["triples"]}]}
                    except json.JSONDecodeError as repair_err:
                        logger.warning(
                            "[RelationExtractor] JSON repair parse failed: msg=%s line=%s col=%s pos=%s",
                            repair_err.msg,
                            repair_err.lineno,
                            repair_err.colno,
                            repair_err.pos,
                        )

        return None

    @staticmethod
    def _repair_json_string(raw_text: str) -> str:
        """Best-effort JSON repair for markdown fences + trailing commas + extra wrappers."""
        text = (raw_text or "").strip()
        if not text:
            return ""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            text = text[start_obj : end_obj + 1]
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

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
                    retry_prompt,
                    response_format="json",
                    priority=LLMPriority.LOW,
                    call_tag="relation_extraction",
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

        logger.debug("[RelationExtractor] extracting relations for facts=%d", len(facts))

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
            result = await self.llm_service.ainvoke(
                prompt,
                response_format="json",
                priority=LLMPriority.LOW,
                call_tag="relation_extraction",
            )
            if isinstance(result, str):
                preview = result[:900].replace("\n", "\\n")
                logger.debug("[RelationExtractor] LLM raw output preview=%s", preview)
            elif isinstance(result, dict):
                preview = json.dumps(result, ensure_ascii=True, default=str)[:900].replace("\n", "\\n")
                logger.debug("[RelationExtractor] LLM raw output preview=%s", preview)

            # Parse with retry logic
            parsed = await self._parse_llm_response(result, prompt, required_format)

            if parsed is None:
                logger.warning("[RelationExtractor] Could not parse LLM response after retries")
                return []

            all_triples: List[Dict[str, Any]] = []
            results_list = parsed.get("results", [])
            raw_triples_count = 0
            parsed_triples_count = 0
            rejected_by_reason: Dict[str, int] = {
                "invalid_index": 0,
                "non_dict_triple": 0,
                "missing_subject": 0,
                "missing_predicate": 0,
                "missing_object": 0,
                "same_entity": 0,
                "low_conf": 0,
                "negation_guard_drop": 0,
                "malformed_triple": 0,
            }

            for item in results_list:
                idx = item.get("index", -1)
                raw_triples = item.get("triples", [])
                raw_triples_count += len(raw_triples or [])

                if idx < 0 or idx >= len(facts):
                    rejected_by_reason["invalid_index"] += len(raw_triples or [])
                    continue

                fact = facts[idx]
                src = fact.get("source_url") or fact.get("source") or None

                for rt in raw_triples:
                    try:
                        if not isinstance(rt, dict):
                            rejected_by_reason["non_dict_triple"] += 1
                            continue
                        parsed_triples_count += 1
                        subj = (rt.get("subject") or "").strip()
                        rel = (rt.get("relation") or "").strip()
                        obj = (rt.get("object") or "").strip()
                        conf = float(rt.get("confidence", 0.0))

                        if not subj:
                            rejected_by_reason["missing_subject"] += 1
                            continue
                        if not rel:
                            rejected_by_reason["missing_predicate"] += 1
                            continue
                        if not obj:
                            rejected_by_reason["missing_object"] += 1
                            continue
                        if subj.strip().lower() == obj.strip().lower():
                            rejected_by_reason["same_entity"] += 1
                            continue
                        if conf < RELATION_MIN_CONF:
                            rejected_by_reason["low_conf"] += 1
                            continue

                        normalized_relation = self._apply_negation_guard(fact.get("statement", ""), rel)
                        if not normalized_relation:
                            rejected_by_reason["negation_guard_drop"] += 1
                            continue

                        triple = {
                            "id": str(uuid.uuid4()),
                            "subject": subj,
                            "relation": normalized_relation,
                            "object": obj,
                            "confidence": max(0.0, min(conf, 1.0)),
                            "source_url": src,
                            "fact_id": fact.get("fact_id"),
                            "source_statement": fact.get("statement", ""),
                        }
                        all_triples.append(triple)
                    except Exception as e:
                        logger.warning(f"[RelationExtractor] Skipping malformed triple: {e}")
                        rejected_by_reason["malformed_triple"] += 1
                        continue

            deduped = dedup_triples_by_structure(all_triples)
            log_value_payload(
                logger,
                "relation_extraction",
                {
                    "raw_triples": raw_triples_count,
                    "parsed_triples": parsed_triples_count,
                    "valid_triples": len(deduped),
                    "rejected_by_reason": rejected_by_reason,
                },
            )
            return deduped

        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[RelationExtractor] Batch extraction failed: {e}")
            return []
        except Exception as e:
            logger.error(f"[RelationExtractor] LLM call failed: {e}")
            return []
