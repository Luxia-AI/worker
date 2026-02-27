import json
import os
import re
import uuid
from hashlib import sha1
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
- Keep only triples directly relevant to the claim context and anchors.
- Prefer intervention/outcome relations that help verify/refute the claim.
- Drop triples that are medically true but unrelated to the claim focus.

IMPORTANT: You MUST respond with valid JSON only. No markdown, no explanations.
Return ONLY valid JSON:
{{"results": [{{"index": 0, "triples": [{{"subject": "...", "relation": "...", "object": "...",
"confidence": 0.9}}]}}]}}
Index integrity:
- Triples under each "index" must come only from that fact index.
- Do not attach triples from one fact to another index.
- If a fact has no claim-relevant triples, return that index with an empty triples list.

ENTITIES PROVIDED: {entities}
CLAIM CONTEXT: {claim_context}

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
        self._generic_anchor_tokens = {
            "health",
            "healthy",
            "disease",
            "diseases",
            "condition",
            "conditions",
            "symptom",
            "symptoms",
            "vitamin",
            "supplement",
            "supplements",
            "immune",
            "immunity",
        }

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

    @classmethod
    def _build_claim_context(
        cls,
        claim_text: str = "",
        claim_entities: Optional[List[str]] = None,
        must_have_entities: Optional[List[str]] = None,
    ) -> str:
        claim = str(claim_text or "").strip()
        claim_entities = [str(e).strip() for e in (claim_entities or []) if str(e).strip()]
        must_have_entities = [str(e).strip() for e in (must_have_entities or []) if str(e).strip()]
        if not claim and not claim_entities and not must_have_entities:
            return "none"
        return (
            f"claim={claim}; "
            f"claim_entities={claim_entities[:10] if claim_entities else []}; "
            f"must_have_entities={must_have_entities[:10] if must_have_entities else []}"
        )

    @classmethod
    def _phrase_tokens(cls, phrase: str) -> set[str]:
        return cls._tokenize(phrase)

    @classmethod
    def _phrase_present(cls, candidate_tokens: set[str], phrase: str) -> bool:
        phrase_toks = cls._phrase_tokens(phrase)
        if not phrase_toks:
            return False
        overlap = len(candidate_tokens & phrase_toks)
        if len(phrase_toks) == 1:
            return overlap >= 1
        # Keep phrase matching strict enough to prevent partial generic drift.
        return overlap >= max(1, len(phrase_toks) - 1)

    def _has_strong_anchor_alignment(
        self,
        candidate_tokens: set[str],
        anchors: Optional[List[str]],
    ) -> bool:
        if not anchors:
            return False
        for anchor in anchors:
            text = str(anchor or "").strip()
            if not text:
                continue
            phrase_toks = self._phrase_tokens(text)
            if not phrase_toks:
                continue
            if len(phrase_toks) == 1 and next(iter(phrase_toks), "") in self._generic_anchor_tokens:
                continue
            if self._phrase_present(candidate_tokens, text):
                return True
        return False

    def _is_claim_aligned_triple(
        self,
        triple: Dict[str, Any],
        fact_statement: str,
        claim_text: str = "",
        claim_entities: Optional[List[str]] = None,
        must_have_entities: Optional[List[str]] = None,
    ) -> bool:
        if not claim_text and not claim_entities and not must_have_entities:
            return True

        triple_tokens = self._tokenize(
            " ".join(
                [
                    str(triple.get("subject") or ""),
                    str(triple.get("relation") or ""),
                    str(triple.get("object") or ""),
                    str(fact_statement or ""),
                ]
            )
        )
        if not triple_tokens:
            return False

        claim_tokens = self._tokenize(claim_text)
        claim_entity_tokens: set[str] = set()
        for ent in claim_entities or []:
            claim_entity_tokens |= self._tokenize(ent)
        must_have_tokens: set[str] = set()
        for ent in must_have_entities or []:
            must_have_tokens |= self._tokenize(ent)

        strong_must_have = self._has_strong_anchor_alignment(triple_tokens, must_have_entities)
        strong_claim_entity = self._has_strong_anchor_alignment(triple_tokens, claim_entities)

        if must_have_tokens and not strong_must_have and len(triple_tokens & must_have_tokens) == 0:
            return False
        if (
            not must_have_tokens
            and claim_entity_tokens
            and not strong_claim_entity
            and len(triple_tokens & claim_entity_tokens) == 0
        ):
            return False
        if claim_tokens and len(triple_tokens & claim_tokens) == 0:
            return False

        claim_pred = self._predicate_family_tokens(claim_text)
        triple_pred = self._predicate_family_tokens(f"{triple.get('relation', '')} {fact_statement or ''}".strip())
        if claim_pred and triple_pred and not (claim_pred & triple_pred):
            lexical_overlap = len(triple_tokens & claim_tokens)
            if lexical_overlap < 2:
                return False
        return True

    def _triple_fact_alignment_score(self, triple: Dict[str, Any], statement: str) -> float:
        stmt_tokens = self._tokenize(statement)
        if not stmt_tokens:
            return 0.0
        subj = str(triple.get("subject") or "")
        rel = str(triple.get("relation") or "")
        obj = str(triple.get("object") or "")
        ent_tokens = self._tokenize(" ".join([subj, obj]))
        rel_tokens = self._tokenize(rel)
        if not ent_tokens and not rel_tokens:
            return 0.0

        ent_overlap = (len(ent_tokens & stmt_tokens) / max(1, len(ent_tokens))) if ent_tokens else 0.0
        rel_overlap = (len(rel_tokens & stmt_tokens) / max(1, len(rel_tokens))) if rel_tokens else 0.0
        phrase_bonus = 0.0
        if subj and self._phrase_present(stmt_tokens, subj):
            phrase_bonus += 0.12
        if obj and self._phrase_present(stmt_tokens, obj):
            phrase_bonus += 0.12
        return max(0.0, min(1.0, (0.75 * ent_overlap) + (0.25 * rel_overlap) + phrase_bonus))

    def _resolve_fact_index(self, idx: int, raw_triple: Dict[str, Any], facts: List[Dict[str, Any]]) -> int:
        if not facts:
            return -1
        try:
            idx_int = int(idx)
        except Exception:
            idx_int = -1

        def _score(i: int) -> float:
            stmt = str(facts[i].get("statement", "") or "")
            return self._triple_fact_alignment_score(raw_triple, stmt)

        # Prefer provided index when it is valid and aligned.
        if 0 <= idx_int < len(facts):
            if _score(idx_int) >= 0.35:
                return idx_int

        # Otherwise remap to best aligned fact.
        best_idx = -1
        best_score = 0.0
        for i in range(len(facts)):
            s = _score(i)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx >= 0 and best_score >= 0.45:
            return best_idx
        return -1

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

    def _try_parse_result(self, result: Any, allow_single_index_fallback: bool = True) -> Optional[Dict[str, Any]]:
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
            if allow_single_index_fallback and "triples" in result:
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
                    if allow_single_index_fallback and "triples" in parsed:
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
                            if allow_single_index_fallback and "triples" in parsed:
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
        self,
        result: Any,
        original_prompt: str,
        required_format: str,
        max_retries: int = 1,
        allow_single_index_fallback: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Parse LLM response with limited retries. NO RECURSION - uses iterative loop."""
        # First attempt: try to parse the original result
        parsed = self._try_parse_result(result, allow_single_index_fallback=allow_single_index_fallback)
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
                parsed = self._try_parse_result(retry_result, allow_single_index_fallback=allow_single_index_fallback)
                if parsed is not None:
                    logger.info(f"[RelationExtractor] Retry {attempt + 1} succeeded")
                    return parsed
            except Exception as e:
                logger.warning(f"[RelationExtractor] Retry {attempt + 1} failed: {e}")

        logger.warning("[RelationExtractor] All retries exhausted, returning None")
        return None

    async def extract_relations(
        self,
        facts: List[Dict[str, Any]],
        entities: List[str],
        claim_text: str = "",
        claim_entities: Optional[List[str]] = None,
        must_have_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
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
        claim_context = self._build_claim_context(
            claim_text=claim_text,
            claim_entities=claim_entities,
            must_have_entities=must_have_entities,
        )
        prompt = (
            BATCH_TRIPLE_PROMPT.format(
                entities=entities_str,
                claim_context=claim_context,
            )
            + facts_text
        )

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
            parsed = await self._parse_llm_response(
                result,
                prompt,
                required_format,
                allow_single_index_fallback=(len(facts) == 1),
            )

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
                "claim_mismatch": 0,
                "malformed_triple": 0,
            }

            claim_hash = ""
            claim_norm = re.sub(r"\s+", " ", str(claim_text or "").strip().lower())
            if claim_norm:
                claim_hash = sha1(claim_norm.encode("utf-8")).hexdigest()[:16]

            for item in results_list:
                idx = item.get("index", -1)
                raw_triples = item.get("triples", [])
                raw_triples_count += len(raw_triples or [])

                # Validate each triple-level index separately below; this block only guards obvious malformed rows.
                if not isinstance(raw_triples, list):
                    rejected_by_reason["invalid_index"] += len(raw_triples or [])
                    continue

                for rt in raw_triples:
                    try:
                        if not isinstance(rt, dict):
                            rejected_by_reason["non_dict_triple"] += 1
                            continue
                        parsed_triples_count += 1
                        resolved_idx = self._resolve_fact_index(idx, rt, facts)
                        if resolved_idx < 0 or resolved_idx >= len(facts):
                            rejected_by_reason["invalid_index"] += 1
                            continue
                        fact = facts[resolved_idx]
                        src = fact.get("source_url") or fact.get("source") or None
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
                            "claim_context_hash": str(fact.get("claim_context_hash") or claim_hash or ""),
                            "claim_context_entities": list(
                                fact.get("claim_context_entities")
                                or fact.get("claim_entities_ctx")
                                or (claim_entities or [])
                            )[:20],
                        }
                        if not self._is_claim_aligned_triple(
                            triple,
                            fact_statement=fact.get("statement", ""),
                            claim_text=claim_text,
                            claim_entities=claim_entities,
                            must_have_entities=must_have_entities,
                        ):
                            rejected_by_reason["claim_mismatch"] += 1
                            continue
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
