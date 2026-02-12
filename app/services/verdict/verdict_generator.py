"""
Verdict Generator - RAG Generation Phase

Takes ranked evidence and generates a final verdict using LLM with augmented facts.
Produces:
- Verdict: TRUE / FALSE / UNVERIFIABLE
- Confidence: 0.0 - 1.0
- Evidence Map: List of supporting/contradicting evidence with relevance
- Rationale: Human-readable explanation

CLAIM-SEGMENT RETRIEVAL:
Before generating verdict, splits claim into segments and retrieves
evidence for each segment independently from VDB. This ensures that
previously-ingested facts are found even when full-claim semantic
similarity favors other topics.
"""

import os
import re
from enum import Enum
from hashlib import sha1
from typing import Any, Dict, List, Optional

from app.constants.config import LLM_TEMPERATURE_VERDICT
from app.core.config import settings
from app.core.logger import get_logger
from app.services.common.claim_segmentation import split_claim_into_segments
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.scraper import Scraper
from app.services.corrective.trusted_search import TrustedSearch
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority
from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy
from app.services.ranking.contradiction_scorer import ContradictionScorer
from app.services.ranking.subclaim_coverage import compute_subclaim_coverage, evaluate_anchor_match
from app.services.ranking.trust_ranker import DummyStanceClassifier
from app.services.retrieval.metadata_enricher import TopicClassifier
from app.services.vdb.vdb_retrieval import VDBRetrieval
from app.services.verdict.policy_override import OverrideSignals, therapeutic_strong_override
from app.services.verdict.rationale_filter import is_efficacy_relevant
from app.shared.anchor_extraction import AnchorExtractor

logger = get_logger(__name__)
_SEGMENT_EVIDENCE_MIN_OVERLAP = float(os.getenv("SEGMENT_EVIDENCE_MIN_OVERLAP", "0.20"))
_POLARITY_DEBUG = os.getenv("VERDICT_DEBUG_POLARITY", "0").strip().lower() in {"1", "true", "yes", "on"}


class Verdict(Enum):
    """Possible verdict outcomes for a claim."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    PARTIALLY_TRUE = "PARTIALLY_TRUE"
    UNVERIFIABLE = "UNVERIFIABLE"


# RAG Prompt for verdict generation
VERDICT_GENERATION_PROMPT = """You are an expert biomedical fact-checker. \
Your task is to evaluate a claim based on the retrieved evidence.

CLAIM TO VERIFY:
{claim}

RETRIEVED EVIDENCE (ranked by relevance and credibility):
{evidence_text}

INSTRUCTIONS:
1. Extract claim segments as short, complete clauses or sentences
2. For each segment, search the evidence for supporting or contradicting facts
3. Determine each segment's status based on evidence strength and credibility
4. Calculate truthfulness_percent by comparing claim content against evidence

CRITICAL RULES FOR CLAIM SEGMENTS:
- claim_segment MUST be copied from the original claim (verbatim), but you may
  trim leading/trailing filler words for readability
- DO NOT paraphrase, summarize, or add new words
- Avoid splitting on a single conjunction unless it forms two independent clauses
- Each segment should be a complete, verifiable statement from the claim

VERDICT OPTIONS:
- TRUE: Evidence strongly supports ALL segments (>=90% truthfulness)
- FALSE: Evidence contradicts the claim (<=30% truthfulness)
- PARTIALLY_TRUE: Some segments supported, others not (30-90% truthfulness)
- UNVERIFIABLE: Insufficient evidence to determine

SEGMENT STATUS OPTIONS:
- VALID: Evidence confirms this exact claim segment
- INVALID: Evidence contradicts this exact claim segment
- PARTIALLY_VALID: Some evidence supports, with minor caveats
- PARTIALLY_INVALID: Some evidence contradicts, not completely wrong
- UNKNOWN: No relevant evidence found for this segment

TRUTHFULNESS CALCULATION:
Analyze the entire claim holistically against all evidence:
- How much of the claim is factually accurate based on evidence?
- Consider evidence quality, source credibility, and direct relevance
- Return as a precise percentage (calculate this based on actual evidence, not example values)
- Example ranges: high accuracy (80-100%), moderate (40-79%), low (0-39%)

CONFIDENCE CALCULATION:
- How confident are you in this verdict based on evidence quality and quantity?
- Consider source credibility, evidence recency, and consensus
- Return as a decimal between 0.0 and 1.0 (calculate this based on actual evidence)

Return ONLY valid JSON (no markdown, no extra text):
{{
    "verdict": "TRUE|FALSE|PARTIALLY_TRUE|UNVERIFIABLE",
    "confidence": 0.XX,
    "truthfulness_percent": XX.X,
    "rationale": "Brief explanation of why this verdict was reached",
    "claim_breakdown": [
        {{
            "claim_segment": "Verbatim text copied from the claim (trim OK)",
            "status": "VALID|INVALID|PARTIALLY_VALID|PARTIALLY_INVALID|UNKNOWN",
            "supporting_fact": "The evidence fact that supports or contradicts this segment",
            "source_url": "https://source.url.of.the.fact"
        }}
    ],
    "evidence_map": [
        {{
            "evidence_id": 0,
            "statement": "The evidence statement",
            "relevance": "SUPPORTS|CONTRADICTS|NEUTRAL",
            "relevance_score": 0.X,
            "source_url": "https://..."
        }}
    ],
    "key_findings": ["Finding 1", "Finding 2"]
}}"""


class VerdictGenerator:
    """
    RAG-based verdict generator using LLM with augmented facts.

    Takes ranked evidence from the pipeline and generates:
    - Final verdict (TRUE/FALSE/PARTIALLY_TRUE/UNVERIFIABLE)
    - Confidence score
    - Evidence map showing how each piece of evidence relates to the claim
    - Human-readable rationale

    CLAIM-SEGMENT RETRIEVAL:
    Before generating verdict, splits the claim into logical segments
    and queries VDB for each segment independently. This ensures facts
    ingested in previous runs are found even when full-claim similarity
    favors other topics (e.g., "vitamin D sunlight" overshadowing
    "vitamin K gut bacteria" facts).
    """

    def __init__(self, vdb_retriever: Optional[VDBRetrieval] = None) -> None:
        self.llm_service = HybridLLMService()
        self.anchor_extractor = AnchorExtractor(self.llm_service)
        self.vdb_retriever = vdb_retriever or VDBRetrieval()
        self.trusted_search = TrustedSearch()
        self.fact_extractor = FactExtractor()
        self.scraper = Scraper()
        self.trust_policy = AdaptiveTrustPolicy()
        self.stance_classifier = DummyStanceClassifier()
        self.contradiction_scorer = ContradictionScorer(semantic_min=0.35)
        self.topic_classifier = TopicClassifier()
        self.MAX_WEB_ROUNDS_PRE_VERDICT = 2
        self.WEB_SEGMENTS_LIMIT = 3
        self.MAX_UNKNOWN_ROUNDS_POST_VERDICT = 2
        env_flag = os.getenv("LUXIA_CONFIDENCE_MODE")
        if env_flag is None:
            self.confidence_mode = bool(getattr(settings, "LUXIA_CONFIDENCE_MODE", False))
        else:
            self.confidence_mode = env_flag.strip().lower() in {"1", "true", "yes", "on"}

    def _quick_web_score(self, segment: str, fact_stmt: str, conf: float) -> float:
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
            "could",
            "would",
            "should",
            "can",
        }
        seg_words = set(w for w in re.findall(r"\b\w+\b", (segment or "").lower()) if w not in stop)
        fact_words = set(w for w in re.findall(r"\b\w+\b", (fact_stmt or "").lower()) if w not in stop)
        overlap = len(seg_words & fact_words)
        overlap_ratio = overlap / max(1, len(seg_words))
        conf = float(conf or 0.0)
        score = 0.25 + 0.45 * max(0.0, min(1.0, conf)) + 0.35 * max(0.0, min(1.0, overlap_ratio))
        return max(0.0, min(1.0, score))

    def _policy_says_insufficient(self, claim: str, evidence: List[Dict[str, Any]]) -> bool:
        if not evidence:
            return True
        trust_policy = getattr(self, "trust_policy", None) or AdaptiveTrustPolicy()

        class _Ev:
            __slots__ = ("statement", "source_url", "semantic_score", "stance", "trust")

            def __init__(self, d: Dict[str, Any]):
                self.statement = d.get("statement") or d.get("text") or ""
                self.source_url = d.get("source_url") or d.get("source") or ""
                self.semantic_score = float(
                    d.get("semantic_score") or d.get("sem_score") or d.get("final_score") or d.get("score") or 0.0
                )
                self.stance = d.get("stance") or "unknown"
                self.trust = float(
                    d.get("trust") or d.get("credibility") or d.get("final_score") or d.get("score") or 0.0
                )

        adapted = [_Ev(d) for d in evidence if (d.get("statement") or d.get("text"))]
        metrics = trust_policy.compute_adaptive_trust(claim, adapted, top_k=min(10, len(adapted)))
        coverage = float(metrics.get("coverage", 0.0) or 0.0)
        num_subclaims = int(metrics.get("num_subclaims", 0) or 0)
        # Keep verdict calibration conservative: unresolved multi-part claims remain insufficient.
        if num_subclaims > 1 and (coverage < 0.99 or len(adapted) < num_subclaims):
            return True
        return not bool(metrics.get("is_sufficient", False))

    def _needs_web_boost(self, evidence: List[Dict[str, Any]], claim: str = "") -> bool:
        """Heuristic: trigger web search when ranked evidence is weak/off-topic."""
        if not evidence:
            return True
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
            "could",
            "would",
            "should",
            "can",
        }
        claim_words = set(w for w in re.findall(r"\b\w+\b", (claim or "").lower()) if w not in stop)

        scores = []
        overlap_ok = False
        for ev in evidence:
            s = ev.get("final_score")
            if s is None:
                s = ev.get("score", 0.0)
            scores.append(float(s or 0.0))

            stmt = (ev.get("statement") or ev.get("text") or "").lower()
            ev_words = set(w for w in re.findall(r"\b\w+\b", stmt) if w not in stop)
            if len(claim_words & ev_words) >= 2:
                overlap_ok = True

        top = max(scores) if scores else 0.0
        avg = (sum(scores) / len(scores)) if scores else 0.0
        return (top < 0.65 or avg < 0.45) or (claim_words and not overlap_ok)

    async def generate_verdict(
        self,
        claim: str,
        ranked_evidence: List[Dict[str, Any]],
        top_k: int = 5,
        used_web_search: bool = False,
        cache_sufficient: bool = False,
        adaptive_metrics: Optional[Dict[str, Any]] = None,
        evidence_snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a verdict for the claim based on ranked evidence.

        ENHANCED: Now performs claim-segment-specific retrieval before
        generating verdict to ensure all relevant facts are found.

        Args:
            claim: The claim/statement to verify
            ranked_evidence: List of ranked evidence from hybrid ranker
            top_k: Number of top evidence items to use (default 5)

        Returns:
            Dict containing:
                - verdict: str (TRUE/FALSE/PARTIALLY_TRUE/UNVERIFIABLE)
                - confidence: float (0-1)
                - rationale: str
                - evidence_map: List of evidence with relevance annotations
                - key_findings: List of key findings
        """
        if self.confidence_mode:
            top_k = max(top_k, int(os.getenv("CONFIDENCE_VERDICT_TOP_K", "10")))
        if not ranked_evidence:
            if used_web_search:
                logger.warning(
                    "[VerdictGenerator] No VDB/KG evidence and web search already used; skipping extra web boost"
                )
                return self._unverifiable_result(claim, "No evidence retrieved (VDB/KG empty after web search)")
            if cache_sufficient:
                logger.info("[VerdictGenerator] Cache sufficient; skipping web boost despite empty evidence")
                return self._unverifiable_result(claim, "Cache marked sufficient but no evidence available")
            logger.warning("[VerdictGenerator] No VDB/KG evidence -> trying web boost")
            segments = self._split_claim_into_segments(claim)[:3]
            web_boost = await self._fetch_web_evidence_for_unknown_segments(segments)
            if not web_boost:
                return self._unverifiable_result(claim, "No evidence retrieved (VDB/KG empty and web boost failed)")
            ranked_evidence = sorted(web_boost, key=self._deterministic_evidence_sort_key)

        # Step 1) Start with VDB/KG evidence + (optional) segment retrieval
        segment_evidence: List[Dict[str, Any]] = []
        if (not cache_sufficient) and (
            self._needs_web_boost(ranked_evidence[: min(len(ranked_evidence), 6)], claim=claim)
            or self._policy_says_insufficient(claim, ranked_evidence[: min(len(ranked_evidence), 10)])
        ):
            segment_evidence = await self._retrieve_segment_evidence(
                claim,
                top_k=3 if self.confidence_mode else 2,
            )

        enriched_evidence = self._merge_evidence(ranked_evidence, segment_evidence)

        if segment_evidence:
            logger.info(
                f"[VerdictGenerator] Evidence: {len(ranked_evidence)} ranked + "
                f"{len(segment_evidence)} segment-specific = {len(enriched_evidence)} total"
            )
        else:
            logger.info(f"[VerdictGenerator] Using {len(ranked_evidence)} ranked evidence (segment retrieval skipped)")

        # Step 2) PRE-VERDICT web-boost loop driven by *sufficiency*
        pre_evidence = enriched_evidence[: min(len(enriched_evidence), max(top_k, 12), 20)]
        if not used_web_search and not cache_sufficient:
            for round_i in range(self.MAX_WEB_ROUNDS_PRE_VERDICT):
                insufficient = self._policy_says_insufficient(claim, pre_evidence)
                weak = self._needs_web_boost(pre_evidence[: min(len(pre_evidence), 6)], claim=claim)
                if not insufficient and not weak:
                    logger.info(f"[VerdictGenerator] Pre-verdict evidence sufficient (round={round_i}). Skipping web.")
                    break
                logger.info(
                    f"[VerdictGenerator] Pre-verdict evidence insufficient/weak (round={round_i}). "
                    f"insufficient={insufficient} weak={weak} -> web search"
                )
                segments = self._split_claim_into_segments(claim)[: self.WEB_SEGMENTS_LIMIT]
                web_boost = await self._fetch_web_evidence_for_unknown_segments(segments)
                if not web_boost:
                    logger.warning("[VerdictGenerator] Web boost returned no facts.")
                    break
                logger.info(f"[VerdictGenerator] Web boost facts: {len(web_boost)}")
                merged_with_web = pre_evidence + web_boost
                merged_with_web.sort(key=self._deterministic_evidence_sort_key)
                pre_evidence = merged_with_web[: min(len(merged_with_web), 18)]
        pre_evidence.sort(key=self._deterministic_evidence_sort_key)
        top_evidence = self._select_balanced_top_evidence(claim, pre_evidence, top_k=min(top_k, 20))

        # Format evidence for prompt
        evidence_text = self._format_evidence_for_prompt(top_evidence)

        # Build RAG prompt
        prompt = VERDICT_GENERATION_PROMPT.format(claim=claim, evidence_text=evidence_text)

        try:
            # HIGH priority - verdict generation is critical
            # Use low temperature for consistent claim segmentation/breakdown
            result = await self.llm_service.ainvoke(
                prompt,
                response_format="json",
                priority=LLMPriority.HIGH,
                temperature=LLM_TEMPERATURE_VERDICT,
                call_tag="verdict_generation",
            )

            # Validate and parse result
            verdict_result = self._parse_verdict_result(
                result,
                claim,
                top_evidence,
                adaptive_metrics=adaptive_metrics,
                evidence_snapshot_id=evidence_snapshot_id,
            )

            logger.info(
                f"[VerdictGenerator] Generated verdict: {verdict_result['verdict']} "
                f"(confidence: {verdict_result['confidence']:.2f})"
            )

            # If UNKNOWN segments remain, run bounded targeted recovery rounds.
            # Efficient strategy: try segment-targeted VDB retrieval first, then
            # do a minimal web pass only if required.
            if not cache_sufficient:
                unknown_round_budget = self.MAX_UNKNOWN_ROUNDS_POST_VERDICT if not used_web_search else 1
                for _ in range(max(1, unknown_round_budget)):
                    unknown_segments = self._get_unknown_segments(verdict_result)
                    if not unknown_segments:
                        break
                    logger.info(
                        "[VerdictGenerator] Found %d UNKNOWN segments, running targeted recovery...",
                        len(unknown_segments),
                    )

                    candidate_boost = await self._retrieve_segment_evidence_for_segments(
                        unknown_segments,
                        top_k=3 if self.confidence_mode else 2,
                    )
                    if candidate_boost:
                        logger.info(
                            "[VerdictGenerator] Retrieved %d targeted facts from VDB for UNKNOWN segments",
                            len(candidate_boost),
                        )
                    else:
                        web_evidence = await self._fetch_web_evidence_for_unknown_segments(
                            unknown_segments,
                            max_queries_per_segment=1 if used_web_search else 2,
                            max_urls_per_query=1 if used_web_search else 3,
                        )
                        candidate_boost = web_evidence
                        if web_evidence:
                            logger.info(
                                "[VerdictGenerator] Retrieved %d additional facts from targeted web search",
                                len(web_evidence),
                            )

                    if not candidate_boost:
                        logger.info(
                            "[VerdictGenerator] No additional targeted evidence found for UNKNOWN segments; stopping."
                        )
                        break

                    enriched_evidence = self._merge_evidence(top_evidence, candidate_boost)
                    top_evidence = self._select_balanced_top_evidence(claim, enriched_evidence, top_k=min(top_k, 20))
                    enriched_evidence_text = self._format_evidence_for_prompt(top_evidence)
                    enriched_prompt = VERDICT_GENERATION_PROMPT.format(
                        claim=claim, evidence_text=enriched_evidence_text
                    )

                    try:
                        enriched_result = await self.llm_service.ainvoke(
                            enriched_prompt,
                            response_format="json",
                            priority=LLMPriority.HIGH,
                            temperature=LLM_TEMPERATURE_VERDICT,
                            call_tag="verdict_generation",
                        )
                        verdict_result = self._parse_verdict_result(
                            enriched_result,
                            claim,
                            top_evidence,
                            adaptive_metrics=adaptive_metrics,
                            evidence_snapshot_id=evidence_snapshot_id,
                        )
                        logger.info(
                            "[VerdictGenerator] Re-generated verdict after targeted recovery: %s (confidence: %.2f)",
                            verdict_result["verdict"],
                            verdict_result["confidence"],
                        )
                    except Exception as e:
                        logger.warning(f"[VerdictGenerator] Failed to re-generate verdict with targeted evidence: {e}")
                        break

            self._log_low_confidence_unverifiable_reason(
                claim=claim,
                ranked_evidence=ranked_evidence,
                final_evidence=top_evidence,
                verdict_result=verdict_result,
            )
            return verdict_result

        except Exception as e:
            logger.error(f"[VerdictGenerator] LLM call failed: {e}")
            return self._unverifiable_result(claim, f"Verdict generation failed: {str(e)}")

    def _format_evidence_for_prompt(self, evidence: List[Dict[str, Any]]) -> str:
        """Format evidence list into readable text for the LLM prompt."""
        lines = []
        for i, ev in enumerate(evidence):
            statement = ev.get("statement") or ev.get("text") or "[No statement]"
            source_url = ev.get("source_url") or ev.get("source") or "Unknown"
            score = ev.get("final_score") or ev.get("score") or 0
            credibility = ev.get("credibility") or 0.5
            grade = ev.get("grade") or "N/A"

            # Skip items without valid statement
            if not statement or statement == "[No statement]":
                continue

            lines.append(
                f"[{i}] Statement: {statement}\n"
                f"    Source: {source_url}\n"
                f"    Score: {score:.2f} | Credibility: {credibility:.2f} | Grade: {grade}"
            )

        return "\n\n".join(lines)

    def _parse_verdict_result(
        self,
        llm_result: Dict[str, Any],
        claim: str,
        evidence: List[Dict[str, Any]],
        adaptive_metrics: Optional[Dict[str, Any]] = None,
        evidence_snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse and validate LLM result, with fallback handling."""
        # Handle LLM error indicators
        if "_llm_error" in llm_result:
            return self._unverifiable_result(claim, llm_result.get("raw_text", "LLM error"))

        # Extract verdict with validation
        verdict_str = llm_result.get("verdict", "UNVERIFIABLE").upper()
        if verdict_str not in [v.value for v in Verdict]:
            verdict_str = "UNVERIFIABLE"
        llm_verdict = verdict_str

        # Extract confidence (will be re-scored from evidence if possible)
        confidence = llm_result.get("confidence", 0.5)
        confidence = max(0.0, min(1.0, float(confidence)))

        # Extract rationale
        rationale = llm_result.get("rationale", "No rationale provided")

        # Extract evidence map or build from evidence
        evidence_map = llm_result.get("evidence_map", [])
        if not evidence_map:
            evidence_map = self._build_default_evidence_map(evidence)
        evidence_map = self._normalize_evidence_map(claim, evidence_map, evidence)

        # Extract key findings
        key_findings = llm_result.get("key_findings", [])

        # Extract claim breakdown for client display
        claim_breakdown = llm_result.get("claim_breakdown", [])
        if self._should_rebuild_claim_breakdown(claim, claim_breakdown):
            claim_breakdown = self._build_deterministic_claim_breakdown(claim, evidence)
        else:
            normalized_breakdown: List[Dict[str, Any]] = []
            seen_segments = set()
            for item in claim_breakdown:
                seg = self._normalize_segment_text(item.get("claim_segment") or "")
                if not self._is_meaningful_segment(seg):
                    continue
                key = seg.lower()
                if key in seen_segments:
                    continue
                seen_segments.add(key)
                item["claim_segment"] = seg
                normalized_breakdown.append(item)
            if normalized_breakdown:
                claim_breakdown = normalized_breakdown
            elif evidence:
                claim_breakdown = self._build_deterministic_claim_breakdown(claim, evidence)
        claim_breakdown = self._align_segments_with_evidence(claim_breakdown, evidence_map, evidence)

        # Guardrail: prevent hallucinated support (source_url / supporting_fact must map to provided evidence)
        ev_urls = set((e.get("source_url") or e.get("source") or "") for e in evidence)
        ev_text = " ".join((e.get("statement") or e.get("text") or "") for e in evidence).lower()
        stop_words = {
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

        def _best_evidence_match(
            segment: str, supporting_fact: str, source_url: str
        ) -> tuple[int, Dict[str, Any] | None, float]:
            seg_tokens = {
                w for w in re.findall(r"\b\w+\b", (segment or "").lower()) if w and w not in stop_words and len(w) > 2
            }
            fact_tokens = {
                w
                for w in re.findall(r"\b\w+\b", (supporting_fact or "").lower())
                if w and w not in stop_words and len(w) > 2
            }
            best_idx = -1
            best_score = -1.0
            best_ev: Dict[str, Any] | None = None
            src_norm = (source_url or "").strip().lower()

            for idx, ev in enumerate(evidence):
                stmt = (ev.get("statement") or ev.get("text") or "").strip()
                if not stmt:
                    continue
                stmt_tokens = {
                    w for w in re.findall(r"\b\w+\b", stmt.lower()) if w and w not in stop_words and len(w) > 2
                }
                src = (ev.get("source_url") or ev.get("source") or "").strip().lower()
                if src_norm and src_norm == src:
                    score = 1.0
                else:
                    seg_overlap = len(seg_tokens & stmt_tokens) / max(1, len(seg_tokens))
                    fact_overlap = len(fact_tokens & stmt_tokens) / max(1, len(fact_tokens)) if fact_tokens else 0.0
                    rel = float(ev.get("final_score") or ev.get("score") or ev.get("sem_score") or 0.0)
                    score = 0.55 * seg_overlap + 0.35 * fact_overlap + 0.10 * rel
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_ev = ev
            return best_idx, best_ev, max(0.0, best_score)

        def _overlap_ok(fact: str, ev_text: str) -> bool:
            if not fact:
                return True
            fw = [w for w in re.findall(r"\b\w+\b", fact.lower()) if w not in stop_words]
            if len(fw) < 3:
                return True  # don't over-penalize short facts
            hits = sum(1 for w in set(fw) if w in ev_text)
            return hits >= 2

        def _has_negation(text: str) -> bool:
            if not text:
                return False
            neg_terms = {
                "no",
                "not",
                "never",
                "none",
                "without",
                "lack",
                "lacks",
                "lacking",
                "myth",
                "debunked",
                "doesn't",
                "isn't",
                "cannot",
                "can't",
            }
            tokens = {w.lower() for w in re.findall(r"\b[\w']+\b", text)}
            return any(t in tokens for t in neg_terms)

        stance_classifier = getattr(self, "stance_classifier", None) or DummyStanceClassifier()
        claim_frame_for_alignment = self._classify_claim_frame(claim)
        therapeutic_efficacy_claim = claim_frame_for_alignment.get("claim_type") == "THERAPEUTIC_EFFICACY"

        for seg in claim_breakdown:
            status = (seg.get("status") or "UNKNOWN").upper()
            seg_text = seg.get("claim_segment") or ""
            src = seg.get("source_url") or ""
            fact = (seg.get("supporting_fact") or "").strip()
            best_idx, best_ev, best_match_score = _best_evidence_match(seg_text, fact, src)
            best_src = ""
            best_stmt = ""
            if best_ev is not None:
                best_src = (best_ev.get("source_url") or best_ev.get("source") or "").strip()
                best_stmt = (best_ev.get("statement") or best_ev.get("text") or "").strip()
                if fact and not src and best_src:
                    seg["source_url"] = best_src
                    src = best_src
                if src and not fact and best_stmt:
                    seg["supporting_fact"] = best_stmt
                    fact = best_stmt
                # If LLM left segment unresolved (empty fact/source), bind best matched evidence.
                if status == "UNKNOWN" and best_match_score >= 0.20:
                    if not fact and best_stmt:
                        seg["supporting_fact"] = best_stmt
                        fact = best_stmt
                    if not src and best_src:
                        seg["source_url"] = best_src
                        src = best_src
                if status != "UNKNOWN":
                    seg["evidence_used_ids"] = [best_idx] if best_idx >= 0 else []
            else:
                seg["evidence_used_ids"] = []
            if (src and src not in ev_urls) or (fact and not _overlap_ok(fact, ev_text)):
                if best_ev is None:
                    seg["status"] = "UNKNOWN"
                    seg["supporting_fact"] = ""
                    seg["source_url"] = ""
                    seg["evidence_used_ids"] = []
                    continue
                # Recover from hallucinated source/fact by re-binding to the best in-pipeline evidence.
                if best_stmt:
                    seg["supporting_fact"] = best_stmt
                    fact = best_stmt
                if best_src:
                    seg["source_url"] = best_src
                    src = best_src
                if best_idx >= 0:
                    seg["evidence_used_ids"] = [best_idx]

            # Polarity guardrails (hard): contradiction cannot be labeled VALID/PARTIALLY_VALID.
            seg_neg = _has_negation(seg_text)
            fact_neg = _has_negation(fact)
            stance = "neutral"
            if best_ev is not None:
                stance = str(best_ev.get("stance") or "neutral")
            if fact:
                try:
                    stance = stance_classifier.classify_stance(seg_text, fact)
                except Exception:
                    stance = stance or "neutral"
            polarity_text = (fact or best_stmt or "").strip()
            if therapeutic_efficacy_claim and polarity_text and not is_efficacy_relevant(polarity_text):
                polarity_text = ""
            polarity = self._segment_polarity(seg_text, polarity_text, stance=stance)
            anchor_overlap = self._segment_anchor_overlap(seg_text, fact or "")
            try:
                best_semantic = float(
                    (best_ev or {}).get("semantic_score")
                    or (best_ev or {}).get("sem_score")
                    or (best_ev or {}).get("final_score")
                    or (best_ev or {}).get("score")
                    or 0.0
                )
            except Exception:
                best_semantic = 0.0
            polarity_neg = _has_negation(polarity_text)
            high_semantic_negation_mismatch = (
                bool(polarity_text) and (seg_neg != polarity_neg) and (best_semantic >= 0.80)
            )

            if status in {"VALID", "PARTIALLY_VALID"} and fact and fact_neg and not seg_neg:
                seg["status"] = "INVALID" if status == "VALID" else "PARTIALLY_INVALID"
            if status in {"VALID", "PARTIALLY_VALID"} and polarity == "contradicts":
                seg["status"] = "INVALID" if status == "VALID" else "PARTIALLY_INVALID"
            if status in {"VALID", "PARTIALLY_VALID"} and high_semantic_negation_mismatch:
                seg["status"] = "INVALID" if status == "VALID" else "PARTIALLY_INVALID"

            # UNKNOWN minimization ladder:
            # deterministically upgrade UNKNOWN only when alignment is strong and polarity is clear.
            if status == "UNKNOWN" and best_ev is not None and polarity_text:
                if not fact:
                    seg["supporting_fact"] = polarity_text
                    fact = polarity_text
                if high_semantic_negation_mismatch:
                    seg["status"] = "INVALID"
                    seg["supporting_fact"] = fact
                    seg["source_url"] = src
                    seg["evidence_used_ids"] = [best_idx] if best_idx >= 0 else []
                elif polarity == "contradicts" and best_semantic >= 0.75:
                    seg["status"] = "PARTIALLY_INVALID"
                    seg["supporting_fact"] = fact
                    seg["source_url"] = src
                    seg["evidence_used_ids"] = [best_idx] if best_idx >= 0 else []
                elif polarity == "entails" and best_semantic >= 0.75:
                    seg["status"] = "PARTIALLY_VALID"
                    seg["supporting_fact"] = fact
                    seg["source_url"] = src
                    seg["evidence_used_ids"] = [best_idx] if best_idx >= 0 else []
                else:
                    deterministic_score = (0.70 * best_match_score) + (0.30 * anchor_overlap)
                    if (
                        deterministic_score >= 0.55
                        and anchor_overlap >= 0.34
                        and polarity in {"entails", "contradicts"}
                    ):
                        seg["status"] = "PARTIALLY_VALID" if polarity == "entails" else "PARTIALLY_INVALID"
                        seg["supporting_fact"] = fact
                        seg["source_url"] = src
                        seg["evidence_used_ids"] = [best_idx] if best_idx >= 0 else []
                    else:
                        seg["status"] = "UNKNOWN"
                        seg["supporting_fact"] = ""
                        seg["source_url"] = ""
                        seg["evidence_used_ids"] = []

        # Evidence quality signal (deterministic, claim-aware) is used for confidence calibration.
        evidence_quality_percent = self._calculate_truthfulness_from_evidence(claim, evidence)

        # Re-score confidence using truth/coverage/agreement/diversity/trust_post calibration.
        diversity_score = 0.0
        if evidence:
            domains = set()
            for ev in evidence[:10]:
                src = ev.get("source_url") or ev.get("source") or ""
                if not src:
                    continue
                try:
                    domain = src.split("/")[2].lower()
                except Exception:
                    continue
                if domain.startswith("www."):
                    domain = domain[4:]
                if domain:
                    domains.add(domain)
            diversity_score = min(1.0, len(domains) / max(1, min(len(evidence), 10)))

        self._log_subclaim_coverage(claim, evidence, claim_breakdown)
        reconciled = self._reconcile_verdict_with_breakdown(claim, claim_breakdown)
        verdict_str = reconciled["verdict"]
        claim_frame = self._classify_claim_frame(claim)
        policy_insufficient = self._policy_says_insufficient(claim, evidence)
        truth_score_percent = self._calculate_truth_score_from_contract(reconciled)
        agreement_ratio = self._calculate_evidence_agreement_ratio(claim, evidence)
        coverage_score = float(reconciled.get("weighted_truth", 0.0) or 0.0)
        adaptive_trust_post = 0.0
        if adaptive_metrics is not None:
            coverage_score = float(adaptive_metrics.get("coverage", coverage_score) or coverage_score)
            agreement_ratio = float(adaptive_metrics.get("agreement", agreement_ratio) or agreement_ratio)
            diversity_score = float(adaptive_metrics.get("diversity", diversity_score) or diversity_score)
            adaptive_trust_post = float(adaptive_metrics.get("trust_post", 0.0) or 0.0)
            logger.info(
                "[VerdictGenerator] Using memoized adaptive trust snapshot=%s trust_post=%.3f",
                (evidence_snapshot_id or "none")[:12],
                adaptive_trust_post,
            )
        elif evidence:
            try:

                class _Ev:
                    __slots__ = ("statement", "source_url", "semantic_score", "stance", "trust")

                    def __init__(self, d: Dict[str, Any]):
                        self.statement = d.get("statement") or d.get("text") or ""
                        self.source_url = d.get("source_url") or d.get("source") or ""
                        self.semantic_score = float(
                            d.get("semantic_score")
                            or d.get("sem_score")
                            or d.get("final_score")
                            or d.get("score")
                            or 0.0
                        )
                        self.stance = d.get("stance") or "unknown"
                        self.trust = float(
                            d.get("trust") or d.get("credibility") or d.get("final_score") or d.get("score") or 0.0
                        )

                adaptive_metrics = self.trust_policy.compute_adaptive_trust(
                    claim,
                    [_Ev(d) for d in evidence if (d.get("statement") or d.get("text"))],
                    top_k=min(10, len(evidence)),
                )
                coverage_score = float(adaptive_metrics.get("coverage", coverage_score) or coverage_score)
                agreement_ratio = float(adaptive_metrics.get("agreement", agreement_ratio) or agreement_ratio)
                diversity_score = float(adaptive_metrics.get("diversity", diversity_score) or diversity_score)
                adaptive_trust_post = float(adaptive_metrics.get("trust_post", 0.0) or 0.0)
            except Exception:
                adaptive_trust_post = 0.0

        if claim_frame.get("is_strong_therapeutic_affirm", False):
            profile = self._therapeutic_evidence_profile(claim, claim_frame, evidence)
            override_verdict, override_reason = therapeutic_strong_override(
                OverrideSignals(
                    high_grade_support=profile["high_grade_support"],
                    high_grade_contra=profile["high_grade_contra"],
                    relevant_noncurative=profile["weak_support"] + profile["scope_mismatch"],
                    relevant_any=(
                        profile["high_grade_support"]
                        + profile["high_grade_contra"]
                        + profile["weak_support"]
                        + profile["scope_mismatch"]
                    ),
                )
            )
            if override_verdict == "MISLEADING":
                misleading = getattr(Verdict, "MISLEADING", None)
                verdict_str = misleading.value if misleading else "MISLEADING"
            elif override_verdict in {Verdict.TRUE.value, Verdict.FALSE.value, Verdict.UNVERIFIABLE.value}:
                verdict_str = override_verdict
            logger.info(
                "[VerdictGenerator][PolicyOverride] type=%s strength=%s polarity=%s subject=%s object=%s "
                "high_grade_support=%d high_grade_contra=%d weak_support=%d scope_mismatch=%d verdict=%s reason=%s",
                claim_frame["claim_type"],
                claim_frame["strength"],
                claim_frame["polarity"],
                claim_frame["subject"] or "-",
                claim_frame["object"] or "-",
                profile["high_grade_support"],
                profile["high_grade_contra"],
                profile["weak_support"],
                profile["scope_mismatch"],
                verdict_str,
                override_reason,
            )

        if (
            float(reconciled.get("weighted_truth", 0.0) or 0.0) >= 0.8
            and int(reconciled.get("strong_covered", 0) or 0) >= 1
            and int(reconciled.get("contradict_count", 0) or 0) == 0
            and agreement_ratio >= 0.8
        ):
            truth_score_percent = max(float(truth_score_percent or 0.0), 85.0)
        ceiling_percent = (0.85 + (0.10 * diversity_score)) * 100.0
        truth_score_percent = min(float(truth_score_percent or 0.0), ceiling_percent)
        truthfulness = max(0.0, min(1.0, float(truth_score_percent or 0.0) / 100.0))
        coverage_score = max(0.0, min(1.0, coverage_score))
        agreement_ratio = max(0.0, min(1.0, agreement_ratio))
        diversity_score = max(0.0, min(1.0, diversity_score))
        adaptive_trust_post = max(0.0, min(1.0, adaptive_trust_post))
        confidence = (
            0.30 * truthfulness
            + 0.25 * coverage_score
            + 0.20 * agreement_ratio
            + 0.15 * diversity_score
            + 0.10 * adaptive_trust_post
        )
        confidence = min(float(confidence), 0.95)
        if coverage_score >= 0.8:
            confidence = max(confidence, truthfulness * 0.75)
        confidence = self._cap_confidence_with_contract(confidence, reconciled, policy_insufficient, verdict_str)
        rationale = self._rewrite_rationale_from_breakdown(rationale, claim_breakdown, reconciled)
        if llm_verdict == Verdict.TRUE.value and verdict_str != Verdict.TRUE.value:
            try:
                truth_score_percent = min(float(truth_score_percent), 89.9)
            except Exception:
                truth_score_percent = 89.9

        return {
            "verdict": verdict_str,
            "confidence": confidence,
            # Backward-compatibility: keep `truthfulness_percent` but now make it a status-driven truth score.
            "truthfulness_percent": truth_score_percent,
            "truth_score_percent": truth_score_percent,
            "evidence_quality_percent": float(evidence_quality_percent or 0.0),
            "rationale": rationale,
            "claim_breakdown": claim_breakdown,
            "evidence_map": evidence_map,
            "key_findings": key_findings,
            "claim": claim,
            "evidence_count": len(evidence),
            "required_segments_count": reconciled["required_segments_count"],
            "resolved_segments_count": reconciled["resolved_segments_count"],
            "required_segments_resolved": reconciled["required_segments_resolved"],
            "unresolved_segments": reconciled["unresolved_segments"],
            "status_weighted_truth": reconciled.get("weighted_truth", 0.0),
            "truthfulness_cap": reconciled.get("truthfulness_cap", 100.0),
            "agreement_ratio": agreement_ratio,
            "policy_sufficient": not policy_insufficient,
            "verdict_reconciled": bool(verdict_str != llm_verdict),
        }

    def _segment_anchor_overlap(self, segment: str, statement: str) -> float:
        eval_result = evaluate_anchor_match(segment, statement)
        groups = eval_result.get("anchor_groups", []) or []
        matched = int(eval_result.get("matched_groups", 0) or 0)
        total = len(groups)
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, matched / total))

    def _calculate_evidence_agreement_ratio(self, claim: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Estimate agreement between claim segments and retrieved evidence.
        Returns a [0..1] ratio where 1.0 means all relevant signals support.
        """
        if not claim or not evidence:
            return 0.0

        segments = self._split_claim_into_segments(claim) or [claim]
        support_signals = 0
        contradict_signals = 0

        for segment in segments:
            best_score = -1.0
            best_polarity = "neutral"

            for ev in evidence[:10]:
                stmt = (ev.get("statement") or ev.get("text") or "").strip()
                if not stmt or not self._segment_topic_guard_ok(segment, stmt):
                    continue

                anchor_eval = evaluate_anchor_match(segment, stmt)
                anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                anchor_ok = bool(anchor_eval.get("anchor_ok", False))
                if not anchor_ok and anchor_overlap < 0.20:
                    continue

                sem_raw = ev.get("semantic_score")
                if sem_raw is None:
                    sem_raw = ev.get("sem_score")
                if sem_raw is None:
                    sem_raw = ev.get("final_score")
                kg_raw = ev.get("kg_score")
                try:
                    sem_score = float(sem_raw or 0.0)
                except Exception:
                    sem_score = 0.0
                try:
                    kg_score = float(kg_raw or 0.0)
                except Exception:
                    kg_score = 0.0
                sem_score = max(0.0, min(1.0, sem_score))
                kg_score = max(0.0, min(1.0, kg_score))
                rel_score = max(0.0, min(1.0, (0.7 * sem_score) + (0.3 * kg_score)))
                combined = (0.75 * rel_score) + (0.25 * anchor_overlap)

                if combined > best_score:
                    stance = str(ev.get("stance", "neutral") or "neutral")
                    best_polarity = self._segment_polarity(segment, stmt, stance=stance)
                    best_score = combined

            if best_score >= 0.20:
                if best_polarity == "entails":
                    support_signals += 1
                elif best_polarity == "contradicts":
                    contradict_signals += 1

        judged = support_signals + contradict_signals
        if judged <= 0:
            return 0.0
        return max(0.0, min(1.0, support_signals / judged))

    @staticmethod
    def _topic_tokens(text: str) -> set[str]:
        weak = {
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
            "do",
            "does",
            "did",
            "not",
            "no",
            "never",
            "cause",
            "causes",
            "caused",
            "link",
            "linked",
            "associate",
            "associated",
        }
        return {w for w in re.findall(r"\b[\w'-]+\b", (text or "").lower()) if len(w) > 2 and w not in weak}

    @staticmethod
    def _is_reporting_statement(text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        reporting_patterns = (
            r"\bclaim(?:s|ed)?\b",
            r"\breport(?:s|ed|ing)?\b",
            r"\baccording to\b",
            r"\breflects?\s+concern\b",
            r"\bsearch(?:ed|ing)?\b",
            r"\bupdated\b",
            r"\bheadline\b",
            r"\brumou?r\b",
        )
        return any(re.search(p, low) for p in reporting_patterns)

    @staticmethod
    def _concept_aliases() -> Dict[str, List[str]]:
        return {
            "vaccine": ["vaccine", "vaccines", "vaccination"],
            "flu": ["flu", "influenza"],
            "autism": ["autism", "asd", "spectrum disorder"],
            "antibiotic": ["antibiotic", "antibiotics", "antibacterial"],
            "cold": ["cold", "colds", "common cold"],
            "virus": ["virus", "viruses", "viral"],
            "sugar": ["sugar", "sucrose", "glucose", "fructose"],
            "hyperactivity": ["hyperactivity", "adhd", "attention deficit"],
            "egg": ["egg", "eggs"],
            "heart": ["heart", "cardiovascular", "cvd", "cholesterol"],
        }

    def _concept_hits(self, text: str) -> set[str]:
        low = (text or "").lower()
        hits: set[str] = set()
        for concept, aliases in self._concept_aliases().items():
            if any(alias in low for alias in aliases):
                hits.add(concept)
        return hits

    @staticmethod
    def _classify_claim_frame(claim: str) -> Dict[str, Any]:
        text = f" {str(claim or '').strip().lower()} "
        therapeutic = bool(re.search(r"\b(cure|cures|treat|treats|therapy|effective against)\b", text))
        strong = bool(re.search(r"\b(cure|cures|eradicate|eliminate)\b", text))
        polarity = "NEGATIVE" if re.search(r"\b(no|not|never|without|does not|do not)\b", text) else "AFFIRM"

        subject = ""
        obj = ""
        m = re.search(r"(?P<subject>.+?)\b(cure|cures|treat|treats|prevent|prevents)\b\s+(?P<object>.+)", text)
        if m:
            subject = re.sub(r"\b(drinking|taking|using|consuming)\b", "", m.group("subject")).strip(" ,.")
            obj = m.group("object").strip(" ,.")
        return {
            "claim_type": "THERAPEUTIC_EFFICACY" if therapeutic else "GENERIC",
            "strength": "STRONG" if strong else "NORMAL",
            "polarity": polarity,
            "subject": subject,
            "object": obj,
            "is_strong_therapeutic_affirm": therapeutic and strong and polarity == "AFFIRM",
        }

    def _therapeutic_evidence_profile(
        self,
        claim: str,
        claim_frame: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        subject = str(claim_frame.get("subject") or "").lower()
        obj = str(claim_frame.get("object") or "").lower()
        high_grade_support = 0
        high_grade_contra = 0
        weak_support = 0
        scope_mismatch = 0
        high_grade_markers = (
            "randomized",
            "systematic review",
            "meta-analysis",
            "cochrane",
            "clinical trial",
            "guideline",
        )
        for ev in evidence[:12]:
            stmt = str(ev.get("statement") or ev.get("text") or "").lower()
            if not stmt:
                continue
            if subject and subject.split()[-1] not in stmt:
                continue
            if obj and obj.split()[-1] not in stmt:
                continue
            stance = str(ev.get("stance") or "neutral")
            pol = self._segment_polarity(claim, stmt, stance=stance)
            sem_raw = ev.get("semantic_score")
            if sem_raw is None:
                sem_raw = ev.get("sem_score")
            if sem_raw is None:
                sem_raw = ev.get("final_score")
            sem = max(0.0, min(1.0, float(sem_raw or 0.0)))
            if sem < 0.55:
                continue
            is_high_grade = any(marker in stmt for marker in high_grade_markers)
            if pol == "contradicts":
                if is_high_grade or sem >= 0.80:
                    high_grade_contra += 1
            elif pol == "entails":
                if is_high_grade:
                    high_grade_support += 1
                else:
                    weak_support += 1
            else:
                scope_mismatch += 1
        return {
            "high_grade_support": high_grade_support,
            "high_grade_contra": high_grade_contra,
            "weak_support": weak_support,
            "scope_mismatch": scope_mismatch,
        }

    def _segment_polarity(self, segment: str, statement: str, stance: str = "neutral") -> str:
        stance_l = (stance or "neutral").lower()
        seg = (segment or "").lower()
        stmt = (statement or "").lower()
        if not seg or not stmt:
            return "neutral"

        def _sanitize_for_negation(text: str) -> str:
            cleaned = re.sub(r"\bnot\s+only\b", " ", text, flags=re.IGNORECASE)
            cleaned = re.sub(r"\bnot\s+necessarily\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\bno\s+longer\b", " ", cleaned, flags=re.IGNORECASE)
            return re.sub(r"\s+", " ", cleaned).strip()

        def _has_semantic_negation(text: str) -> bool:
            low = _sanitize_for_negation(text or "")
            if not low:
                return False
            strong_patterns = (
                r"\b(?:ineffective|inefficacy|not effective|not recommended|no effect|cannot treat|can't treat)\b",
                r"\b(?:do|does|did)\s+not\s+(?:work|treat|help|cure|prevent|cause|reduce)\b",
                r"\b(?:doesn't|don't|didn't)\s+(?:work|treat|help|cure|prevent|cause|reduce)\b",
                r"\b(?:won't|will not|cannot|can't)\s+(?:work|treat|help|cure|prevent|cause|reduce)\b",
                r"\b(?:not|no|never)\s+(?:work|treat|help|cure|effective|efficacious)\b",
                r"\b(?:fails?\s+to|unable\s+to)\s+(?:work|treat|help|cure|prevent|reduce)\b",
            )
            if any(re.search(p, low, flags=re.IGNORECASE) for p in strong_patterns):
                return True
            return bool(
                re.search(
                    r"\b(?:is|are|was|were|can|could|may|might|must|should|would|will|has|have|had)\s+not\b",
                    low,
                    flags=re.IGNORECASE,
                )
            )

        def _predicate_groups(text: str) -> set[str]:
            groups = {
                "efficacy": (
                    r"\b(?:work|works|working|treat|treats|treated|treating|cure|cures|cured|curing|"
                    r"help|helps|helped|effective|efficacy|ineffective|inefficacy|efficacious)\b"
                ),
                "causal": (
                    r"\b(?:cause|causes|caused|causing|link|linked|associate|associated|"
                    r"result|results|resulted|lead|leads|leading)\b"
                ),
                "prevention": (
                    r"\b(?:prevent|prevents|prevented|prevention|reduce|reduces|reduced|reduction|"
                    r"protect|protects|protected)\b"
                ),
            }
            hits: set[str] = set()
            for group, pattern in groups.items():
                if re.search(pattern, text, flags=re.IGNORECASE):
                    hits.add(group)
            return hits

        seg_neg = _has_semantic_negation(seg)
        stmt_neg = _has_semantic_negation(stmt)
        seg_groups = _predicate_groups(seg)
        stmt_groups = _predicate_groups(stmt)
        same_predicate = bool(seg_groups & stmt_groups)

        # Negation symmetry guard (runs before stance fallback):
        # negative + negative over same predicate => support, not contradiction.
        if same_predicate and seg_neg and stmt_neg:
            if _POLARITY_DEBUG:
                logger.info(
                    "[VerdictGenerator][Polarity] seg_neg=%s stmt_neg=%s same_predicate=%s stance=%s => entails",
                    seg_neg,
                    stmt_neg,
                    same_predicate,
                    stance_l,
                )
            return "entails"
        # Same predicate with polarity mismatch => contradiction.
        if same_predicate and (seg_neg ^ stmt_neg):
            if _POLARITY_DEBUG:
                logger.info(
                    "[VerdictGenerator][Polarity] seg_neg=%s stmt_neg=%s same_predicate=%s stance=%s => contradicts",
                    seg_neg,
                    stmt_neg,
                    same_predicate,
                    stance_l,
                )
            return "contradicts"

        if stance_l in {"entails", "contradicts"}:
            if _POLARITY_DEBUG:
                logger.info(
                    "[VerdictGenerator][Polarity] seg_neg=%s stmt_neg=%s same_predicate=%s stance=%s => %s",
                    seg_neg,
                    stmt_neg,
                    same_predicate,
                    stance_l,
                    stance_l,
                )
            return stance_l

        neg_causal_re = re.compile(
            (
                r"\b(?:do|does|did|can|could|may|might|must|should|would|will|is|are|was|were)?\s*"
                r"(?:not|never|no)\s+(?:cause|causes|caused|causing|link(?:ed)?|associate(?:d)?|"
                r"result(?:s|ed|ing)?\s+in|lead(?:s|ing)?\s+to)\b"
            ),
            flags=re.IGNORECASE,
        )
        pos_causal_re = re.compile(
            (
                r"\b(?:cause|causes|caused|causing|contribut(?:e|es|ed|ing)\s+to|"
                r"link(?:ed)?\s+to|associate(?:d)?\s+with|lead(?:s|ing)?\s+to|"
                r"result(?:s|ed|ing)?\s+in)\b"
            ),
            flags=re.IGNORECASE,
        )
        hedge_pos_re = re.compile(
            r"\b(?:may|might|can|could|possibly|suggest(?:s|ed)?)\b.{0,25}\b(?:cause|linked?|associate)\b",
            flags=re.IGNORECASE,
        )
        no_link_re = re.compile(r"\bno\s+link\b|\bnot\s+associated\b", flags=re.IGNORECASE)

        seg_neg = bool(neg_causal_re.search(seg) or no_link_re.search(seg) or seg_neg)
        stmt_neg = bool(neg_causal_re.search(stmt) or no_link_re.search(stmt) or stmt_neg)
        seg_pos = bool(pos_causal_re.search(seg)) and not seg_neg
        stmt_pos = bool(pos_causal_re.search(stmt)) and not stmt_neg
        stmt_hedged_pos = bool(hedge_pos_re.search(stmt)) and not stmt_neg

        if seg_neg and (stmt_pos or stmt_hedged_pos) and not stmt_neg:
            return "contradicts"
        if seg_pos and stmt_neg:
            return "contradicts"
        if seg_neg and stmt_neg:
            return "entails"
        if seg_pos and stmt_pos and not stmt_hedged_pos:
            return "entails"
        if _POLARITY_DEBUG:
            logger.info(
                "[VerdictGenerator][Polarity] seg_neg=%s stmt_neg=%s same_predicate=%s stance=%s => neutral",
                seg_neg,
                stmt_neg,
                same_predicate,
                stance_l,
            )
        return "neutral"

    def _segment_topic_guard_ok(self, segment: str, statement: str) -> bool:
        seg_concepts = self._concept_hits(segment)
        stmt_concepts = self._concept_hits(statement)
        if {"vaccine", "flu"}.issubset(seg_concepts) and not {"vaccine", "flu"}.issubset(stmt_concepts):
            return False
        if "antibiotic" in seg_concepts and ({"cold", "flu", "virus"} & seg_concepts):
            if "antibiotic" not in stmt_concepts:
                return False
            if not ({"cold", "flu", "virus"} & stmt_concepts):
                return False
        if {"sugar", "hyperactivity"}.issubset(seg_concepts) and not {"sugar", "hyperactivity"}.issubset(stmt_concepts):
            return False

        seg_tokens = self._topic_tokens(segment)
        stmt_tokens = self._topic_tokens(statement)
        if len(seg_tokens) >= 3 and len(seg_tokens & stmt_tokens) == 0 and not (seg_concepts & stmt_concepts):
            return False
        return True

    def _evidence_score(self, ev: Dict[str, Any]) -> float:
        score = ev.get("final_score")
        if score is None:
            score = ev.get("score")
        if score is None:
            score = ev.get("sem_score")
        try:
            score_f = float(score or 0.0)
        except Exception:
            score_f = 0.0
        statement = str(ev.get("statement") or ev.get("text") or "")
        if self._is_reporting_statement(statement):
            score_f *= 0.40
        return score_f

    def _select_balanced_top_evidence(
        self, claim: str, evidence: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        if not evidence:
            return []
        candidates = [ev for ev in evidence if (ev.get("statement") or ev.get("text"))]
        if not candidates:
            return []
        candidates = sorted(candidates, key=self._deterministic_evidence_sort_key)

        segments = self._split_claim_into_segments(claim)
        selected: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _add(ev: Dict[str, Any]) -> None:
            key = self._normalize_statement_key(str(ev.get("statement") or ev.get("text") or ""))
            if key and key not in seen:
                seen.add(key)
                selected.append(ev)

        # First pass: guarantee per-segment representation when possible.
        for segment in segments:
            best_ev: Optional[Dict[str, Any]] = None
            best_score = -1.0
            for ev in candidates:
                if str(ev.get("stance", "neutral") or "neutral").lower() == "contradicts":
                    continue
                stmt = str(ev.get("statement") or ev.get("text") or "")
                if not self._segment_topic_guard_ok(segment, stmt):
                    continue
                anchor_eval = evaluate_anchor_match(segment, stmt)
                if not bool(anchor_eval.get("anchor_ok", False)):
                    continue
                score = (0.75 * self._evidence_score(ev)) + (
                    0.25 * float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                )
                if score > best_score:
                    best_score = score
                    best_ev = ev
            if best_ev is not None:
                _add(best_ev)
            if len(selected) >= top_k:
                return selected[:top_k]

        # Second pass: fill with strongest non-contradicting evidence.
        for ev in candidates:
            if str(ev.get("stance", "neutral") or "neutral").lower() == "contradicts":
                continue
            _add(ev)
            if len(selected) >= top_k:
                break

        # Fallback: if everything is contradicting, use deterministic top-ranked.
        if not selected:
            for ev in candidates:
                _add(ev)
                if len(selected) >= top_k:
                    break

        return selected[:top_k]

    def _normalize_evidence_map(
        self,
        claim: str,
        evidence_map: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        evidence_by_id: Dict[int, Dict[str, Any]] = {idx: ev for idx, ev in enumerate(evidence)}

        for item in evidence_map or []:
            if not isinstance(item, dict):
                continue
            ev_id = item.get("evidence_id")
            try:
                ev_idx = int(ev_id)
            except Exception:
                ev_idx = -1
            ev = evidence_by_id.get(ev_idx, {})
            statement = (item.get("statement") or ev.get("statement") or ev.get("text") or "").strip()
            source_url = (item.get("source_url") or ev.get("source_url") or ev.get("source") or "").strip()
            relevance_score = float(item.get("relevance_score", ev.get("final_score", ev.get("score", 0.0))) or 0.0)
            anchor_score = float(ev.get("anchor_match_score", self._segment_anchor_overlap(claim, statement)) or 0.0)
            relevance = str(item.get("relevance", "NEUTRAL") or "NEUTRAL").upper()
            if anchor_score < 0.2:
                relevance = "NEUTRAL"
                relevance_score *= 0.6
            if self._is_reporting_statement(statement):
                relevance = "NEUTRAL"
                relevance_score *= 0.35
            normalized.append(
                {
                    "evidence_id": ev_idx if ev_idx >= 0 else len(normalized),
                    "statement": statement,
                    "relevance": relevance,
                    "relevance_score": max(0.0, min(1.0, relevance_score)),
                    "source_url": source_url,
                    "anchor_match_score": anchor_score,
                }
            )
        return normalized

    def _align_segments_with_evidence(
        self,
        claim_breakdown: List[Dict[str, Any]],
        evidence_map: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not claim_breakdown:
            return claim_breakdown

        evidence_by_id: Dict[int, Dict[str, Any]] = {idx: ev for idx, ev in enumerate(evidence)}
        allowed_rel = {"SUPPORTS", "CONTRADICTS", "PARTIAL", "PARTIALLY_SUPPORTS", "PARTIALLY_CONTRADICTS"}

        for seg in claim_breakdown:
            segment = (seg.get("claim_segment") or "").strip()
            best_item: Dict[str, Any] | None = None
            best_score = -1.0
            for em in evidence_map:
                rel = str(em.get("relevance", "NEUTRAL") or "NEUTRAL").upper()
                if rel not in allowed_rel:
                    continue
                statement = (em.get("statement") or "").strip()
                if not statement:
                    continue
                if not self._segment_topic_guard_ok(segment, statement):
                    continue
                anchor_eval = evaluate_anchor_match(segment, statement)
                if not bool(anchor_eval.get("anchor_ok", False)):
                    continue
                anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                if anchor_overlap < _SEGMENT_EVIDENCE_MIN_OVERLAP:
                    continue
                rel_score = float(em.get("relevance_score", 0.0) or 0.0)
                score = (0.8 * rel_score) + (0.2 * anchor_overlap)
                if score > best_score:
                    best_item = em
                    best_score = score

            if best_item is None:
                # Fallback to segment-retrieved evidence pool.
                for idx, ev in enumerate(evidence):
                    seg_q = (ev.get("_segment_query") or "").strip().lower()
                    statement = (ev.get("statement") or ev.get("text") or "").strip()
                    if not statement:
                        continue
                    if seg_q and seg_q not in segment.lower():
                        continue
                    if not self._segment_topic_guard_ok(segment, statement):
                        continue
                    anchor_eval = evaluate_anchor_match(segment, statement)
                    if not bool(anchor_eval.get("anchor_ok", False)):
                        continue
                    anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                    if anchor_overlap < _SEGMENT_EVIDENCE_MIN_OVERLAP:
                        continue
                    score = (0.7 * float(ev.get("final_score", ev.get("score", 0.0)) or 0.0)) + (0.3 * anchor_overlap)
                    if score > best_score:
                        best_score = score
                        best_item = {
                            "evidence_id": idx,
                            "statement": statement,
                            "source_url": ev.get("source_url") or ev.get("source") or "",
                            "relevance_score": float(ev.get("final_score", ev.get("score", 0.0)) or 0.0),
                        }

            if best_item is not None:
                ev_id = int(best_item.get("evidence_id", -1) or -1)
                ev = evidence_by_id.get(ev_id, {})
                statement = (best_item.get("statement") or ev.get("statement") or ev.get("text") or "").strip()
                source_url = (best_item.get("source_url") or ev.get("source_url") or ev.get("source") or "").strip()
                if statement:
                    seg["supporting_fact"] = statement
                if statement and source_url:
                    seg["source_url"] = source_url
                seg["evidence_used_ids"] = [ev_id] if ev_id >= 0 else []
                seg["alignment_debug"] = {
                    "reason": "mapped_from_evidence_map",
                    "anchor_overlap": round(self._segment_anchor_overlap(segment, statement), 3),
                    "score": round(best_score, 3),
                }
            else:
                seg.setdefault("evidence_used_ids", [])
                seg["status"] = "UNKNOWN"
                seg["supporting_fact"] = ""
                seg["source_url"] = ""
                seg["alignment_debug"] = {
                    "reason": "no_relevant_evidence",
                    "min_overlap": _SEGMENT_EVIDENCE_MIN_OVERLAP,
                }
        return claim_breakdown

    def _log_subclaim_coverage(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
        claim_breakdown: List[Dict[str, Any]],
    ) -> None:
        trust_policy = getattr(self, "trust_policy", None) or AdaptiveTrustPolicy()
        subclaims = trust_policy.decompose_claim(claim)
        anchor_extractor = getattr(self, "anchor_extractor", None)
        if anchor_extractor is None:
            anchor_extractor = AnchorExtractor()
            self.anchor_extractor = anchor_extractor
        anchor_result = anchor_extractor.extract_for_claim(claim=claim, subclaims=subclaims, entity_hints=[])
        cov = compute_subclaim_coverage(
            subclaims,
            evidence,
            partial_weight=0.5,
            anchors_by_subclaim=anchor_result.anchors_by_subclaim,
        )
        details = cov.get("details", [])
        if not details:
            logger.info("[VerdictGenerator][Coverage] subclaims=0 covered=0 strong=0 partial=0 unknown=0 coverage=0.00")
            return

        strong = 0
        partial = 0
        invalid = 0
        unknown = 0
        for d in details:
            status = (d.get("status") or "UNKNOWN").upper()
            if status == "STRONGLY_VALID":
                strong += 1
            elif status == "PARTIALLY_VALID":
                partial += 1
            elif status in {"INVALID", "PARTIALLY_INVALID"}:
                invalid += 1
            else:
                unknown += 1
            logger.info(
                "[VerdictGenerator][Coverage] subclaim=%d status=%s best_evidence_id=%s "
                "semantic=%.3f relevance=%.3f overlap=%.3f anchors=%d/%d terms=%s segment=%s",
                d.get("subclaim_id"),
                status,
                d.get("best_evidence_id"),
                float(d.get("semantic_score", 0.0)),
                float(d.get("relevance_score", 0.0)),
                float(d.get("overlap", 0.0)),
                int(d.get("anchors_matched", 0)),
                int(d.get("anchors_required", 0)),
                d.get("anchors", []),
                (d.get("subclaim") or "")[:80],
            )

        logger.info(
            "[VerdictGenerator][Coverage] subclaims=%d weighted_covered=%.2f strong=%d partial=%d invalid=%d "
            "unknown=%d coverage=%.2f",
            int(cov.get("subclaims", len(details))),
            float(cov.get("weighted_covered", 0.0)),
            strong,
            partial,
            invalid,
            unknown,
            float(cov.get("coverage", 0.0)),
        )
        try:

            class _Ev:
                __slots__ = ("statement", "source_url", "semantic_score", "stance", "trust")

                def __init__(self, d: Dict[str, Any]):
                    self.statement = d.get("statement") or d.get("text") or ""
                    self.source_url = d.get("source_url") or d.get("source") or ""
                    self.semantic_score = float(
                        d.get("semantic_score") or d.get("sem_score") or d.get("final_score") or d.get("score") or 0.0
                    )
                    self.stance = d.get("stance") or "unknown"
                    self.trust = float(d.get("trust") or d.get("final_score") or d.get("score") or 0.0)

            adaptive = trust_policy.compute_adaptive_trust(
                claim, [_Ev(d) for d in evidence if (d.get("statement") or d.get("text"))], top_k=min(12, len(evidence))
            )
            logger.info(
                "[VerdictGenerator][Coverage][Aligned] verdict_coverage=%.2f adaptive_coverage=%.2f",
                float(cov.get("coverage", 0.0)),
                float(adaptive.get("coverage", 0.0)),
            )
        except Exception as e:
            logger.warning("[VerdictGenerator][Coverage] adaptive alignment failed: %s", e)

    def _is_meaningful_segment(self, segment: str) -> bool:
        if not segment:
            return False
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
        words = [w for w in re.findall(r"\b\w+\b", segment.lower()) if w not in stop]
        return len(words) >= 2 and len(segment.strip()) >= 12

    def _should_rebuild_claim_breakdown(self, claim: str, claim_breakdown: List[Dict[str, Any]]) -> bool:
        if not claim_breakdown:
            return True
        required_segments = self._split_claim_into_segments(claim)
        required_count = len(required_segments) if required_segments else 1
        unknown_count = 0
        low_quality = 0
        normalized_meaningful_count = 0
        for item in claim_breakdown:
            seg = self._normalize_segment_text(item.get("claim_segment") or "")
            if not self._is_meaningful_segment(seg):
                low_quality += 1
            else:
                normalized_meaningful_count += 1
            if (item.get("status") or "UNKNOWN").upper() == "UNKNOWN":
                unknown_count += 1
        if low_quality > 0:
            return True
        if normalized_meaningful_count != required_count:
            return True
        return unknown_count >= max(1, int(0.8 * len(claim_breakdown)))

    def _normalize_segment_text(self, segment: str) -> str:
        """Normalize duplicated prefixes (e.g., 'A diet A diet rich in ...')."""
        s = re.sub(r"\s+", " ", (segment or "")).strip(" ,.").strip()
        if not s:
            return ""
        words = s.split()
        for n in (3, 2, 1):
            if len(words) >= 2 * n:
                left = " ".join(words[:n]).lower()
                right = " ".join(words[n : (2 * n)]).lower()
                if left == right:
                    s = " ".join(words[n:])
                    break
        if s.lower().startswith("a diet a diet "):
            s = s[len("A diet ") :]
        return re.sub(r"\s+", " ", s).strip(" ,.").strip()

    def _match_required_segment_statuses(self, claim: str, claim_breakdown: List[Dict[str, Any]]) -> List[str]:
        required_segments = self._split_claim_into_segments(claim)
        if not required_segments:
            required_segments = [re.sub(r"\s+", " ", (claim or "")).strip(" ,.")]

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

        def _tokens(text: str) -> set[str]:
            return {w for w in re.findall(r"\b[\w']+\b", (text or "").lower()) if w and w not in stop and len(w) > 2}

        normalized_breakdown: List[Dict[str, Any]] = []
        for item in claim_breakdown or []:
            seg = self._normalize_segment_text(item.get("claim_segment") or "")
            if not self._is_meaningful_segment(seg):
                continue
            normalized_breakdown.append(
                {
                    "segment": seg,
                    "tokens": _tokens(seg),
                    "status": (item.get("status") or "UNKNOWN").upper(),
                }
            )

        matched_statuses: List[str] = []
        used = set()
        for req in required_segments:
            req_tokens = _tokens(req)
            best_idx = -1
            best_overlap = 0.0
            for idx, cand in enumerate(normalized_breakdown):
                if idx in used:
                    continue
                overlap = len(req_tokens & cand["tokens"]) / max(1, len(req_tokens))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            if best_idx >= 0 and best_overlap >= 0.30:
                used.add(best_idx)
                matched_statuses.append(normalized_breakdown[best_idx]["status"])
            else:
                matched_statuses.append("UNKNOWN")
        return matched_statuses

    @staticmethod
    def _status_truth_weight(status: str) -> float:
        weights = {
            "VALID": 1.0,
            "PARTIALLY_VALID": 0.65,
            "PARTIALLY_INVALID": 0.35,
            "INVALID": 0.0,
            "UNKNOWN": 0.0,
        }
        return float(weights.get((status or "UNKNOWN").upper(), 0.0))

    def _status_contract(self, claim: str, claim_breakdown: List[Dict[str, Any]]) -> Dict[str, Any]:
        statuses = self._match_required_segment_statuses(claim, claim_breakdown)
        total = len(statuses)
        unresolved = sum(1 for s in statuses if (s or "").upper() == "UNKNOWN")
        resolved = total - unresolved
        weighted_truth = sum(self._status_truth_weight(s) for s in statuses) / max(1, total)
        weighted_truth = max(0.0, min(1.0, weighted_truth))
        resolved_ratio = resolved / max(1, total)
        has_support = any((s or "").upper() in {"VALID", "PARTIALLY_VALID"} for s in statuses)
        has_invalid = any((s or "").upper() in {"INVALID", "PARTIALLY_INVALID"} for s in statuses)
        strong_covered = sum(1 for s in statuses if (s or "").upper() in {"VALID", "STRONGLY_VALID"})
        contradict_count = sum(1 for s in statuses if (s or "").upper() in {"INVALID", "PARTIALLY_INVALID"})
        truthfulness_cap = min(100.0, (weighted_truth * 100.0) + 5.0)
        if unresolved > 0:
            truthfulness_cap = min(truthfulness_cap, 65.0)
        if resolved == 0:
            truthfulness_cap = min(truthfulness_cap, 20.0)
        return {
            "statuses": statuses,
            "required_segments_count": total,
            "resolved_segments_count": resolved,
            "required_segments_resolved": unresolved == 0,
            "unresolved_segments": unresolved,
            "weighted_truth": weighted_truth,
            "resolved_ratio": resolved_ratio,
            "truthfulness_cap": max(0.0, truthfulness_cap),
            "has_support": has_support,
            "has_invalid": has_invalid,
            "strong_covered": strong_covered,
            "contradict_count": contradict_count,
        }

    def _calculate_truth_score_from_contract(self, reconciled: Dict[str, Any]) -> float:
        """
        Truth Score reflects "how true the claim is" from segment outcomes, not evidence strength.
        Confidence is handled separately using evidence quality metrics.
        """
        statuses = [str(s or "UNKNOWN").upper() for s in (reconciled.get("matched_statuses") or [])]
        if not statuses:
            return 0.0

        weights = {
            "STRONGLY_VALID": 1.0,
            "VALID": 1.0,
            "PARTIALLY_VALID": 0.75,
            "UNKNOWN": 0.45,
            "PARTIALLY_INVALID": 0.25,
            "INVALID": 0.0,
        }
        total = len(statuses)
        base = sum(float(weights.get(s, 0.0)) for s in statuses) / max(1, total)

        contradict_count = int(reconciled.get("contradict_count", 0) or 0)
        unresolved = int(reconciled.get("unresolved_segments", 0) or 0)
        strong_covered = int(reconciled.get("strong_covered", 0) or 0)
        required_segments = int(reconciled.get("required_segments_count", total) or total)
        has_support = bool(reconciled.get("has_support", False))
        has_invalid = bool(reconciled.get("has_invalid", False))
        verdict = str(reconciled.get("verdict", Verdict.UNVERIFIABLE.value) or Verdict.UNVERIFIABLE.value)

        contradict_ratio = contradict_count / max(1, total)
        unresolved_ratio = unresolved / max(1, total)

        score = base
        score -= 0.30 * contradict_ratio
        score -= 0.10 * unresolved_ratio

        # Keep unresolved claims from inflating too high.
        if unresolved > 0 and contradict_count == 0 and has_support:
            score = min(score, 0.89)

        # Strong fully-resolved support should read as near-certain truth.
        if (
            verdict == Verdict.TRUE.value
            and unresolved == 0
            and contradict_count == 0
            and strong_covered >= required_segments
            and base >= 0.99
        ):
            score = max(score, 0.98)

        # Pure contradiction should be clearly low.
        if verdict == Verdict.FALSE.value and has_invalid and not has_support:
            score = min(score, 0.12)

        score = max(0.0, min(1.0, score))
        return round(score * 100.0, 1)

    def _cap_confidence_with_contract(
        self,
        confidence: float,
        contract: Dict[str, Any],
        policy_insufficient: bool,
        verdict: str,
    ) -> float:
        cap = 0.98
        unresolved = int(contract.get("unresolved_segments", 0) or 0)
        resolved_ratio = float(contract.get("resolved_ratio", 0.0) or 0.0)
        weighted_truth = float(contract.get("weighted_truth", 0.0) or 0.0)
        if unresolved > 0:
            cap = min(cap, 0.40 + (0.35 * resolved_ratio))
        if policy_insufficient:
            cap = min(cap, 0.62 if unresolved == 0 else 0.55)
        if verdict == Verdict.UNVERIFIABLE.value:
            cap = min(cap, 0.35)
        if unresolved > 0 and weighted_truth <= 0.35:
            cap = min(cap, 0.48)
        return max(0.05, min(float(confidence or 0.0), cap))

    def _reconcile_verdict_with_breakdown(self, claim: str, claim_breakdown: List[Dict[str, Any]]) -> Dict[str, Any]:
        contract = self._status_contract(claim, claim_breakdown)
        statuses = contract["statuses"]
        required_segments_count = int(contract["required_segments_count"] or 0)
        unresolved_segments = int(contract["unresolved_segments"] or 0)
        resolved_segments_count = int(contract["resolved_segments_count"] or 0)
        weighted_truth = float(contract.get("weighted_truth", 0.0) or 0.0)
        strong_covered = int(contract.get("strong_covered", 0) or 0)
        contradict_count = int(contract.get("contradict_count", 0) or 0)
        has_support = bool(contract["has_support"])
        has_invalid = bool(contract["has_invalid"])
        all_valid = bool(statuses) and all(s == "VALID" for s in statuses)
        all_invalid = bool(statuses) and all(s == "INVALID" for s in statuses)

        if unresolved_segments > 0:
            verdict = Verdict.PARTIALLY_TRUE.value if (has_support or has_invalid) else Verdict.UNVERIFIABLE.value
        elif all_valid:
            verdict = Verdict.TRUE.value
        elif all_invalid:
            verdict = Verdict.FALSE.value
        else:
            verdict = Verdict.PARTIALLY_TRUE.value

        # Safety guard: do not emit FALSE when there is support and no contradiction.
        if verdict == Verdict.FALSE.value and has_support and contradict_count == 0:
            verdict = Verdict.PARTIALLY_TRUE.value

        # Narrow override for fully resolved, strongly supported outcomes.
        if (
            unresolved_segments == 0
            and weighted_truth >= 0.8
            and strong_covered >= required_segments_count
            and contradict_count == 0
        ):
            verdict = Verdict.TRUE.value

        return {
            "verdict": verdict,
            "required_segments_count": required_segments_count,
            "resolved_segments_count": resolved_segments_count,
            "required_segments_resolved": unresolved_segments == 0,
            "unresolved_segments": unresolved_segments,
            "matched_statuses": statuses,
            "weighted_truth": contract["weighted_truth"],
            "truthfulness_cap": contract["truthfulness_cap"],
            "resolved_ratio": contract["resolved_ratio"],
            "has_support": has_support,
            "has_invalid": has_invalid,
        }

    def _rewrite_rationale_from_breakdown(
        self,
        original_rationale: str,
        claim_breakdown: List[Dict[str, Any]],
        reconciled: Dict[str, Any],
    ) -> str:
        statuses = [str(item.get("status", "UNKNOWN") or "UNKNOWN").upper() for item in claim_breakdown]
        valid_count = sum(1 for s in statuses if s in {"VALID", "PARTIALLY_VALID"})
        invalid_count = sum(1 for s in statuses if s in {"INVALID", "PARTIALLY_INVALID"})
        unknown_count = sum(1 for s in statuses if s == "UNKNOWN")
        total = max(1, len(statuses))
        verdict = str(reconciled.get("verdict", Verdict.UNVERIFIABLE.value))
        unresolved = int(reconciled.get("unresolved_segments", unknown_count) or 0)

        if unresolved > 0 or unknown_count > 0:
            return (
                f"Evidence supports {valid_count}/{total} segment(s), "
                f"while {unknown_count} segment(s) remain unresolved. "
                f"Verdict is {verdict} until segment-level evidence is complete."
            )
        if invalid_count > 0 and valid_count > 0:
            return (
                f"Evidence is mixed: {valid_count}/{total} segment(s) are supported and "
                f"{invalid_count}/{total} are contradicted."
            )
        if invalid_count == total:
            return "Evidence contradicts all required claim segments."
        if valid_count == total and verdict == Verdict.TRUE.value:
            return original_rationale or "Evidence supports all required claim segments."
        return original_rationale or "Verdict is based on segment-level evidence evaluation."

    def _build_deterministic_claim_breakdown(self, claim: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build a deterministic claim breakdown with meaningful segments.
        Used when LLM returns fragmentary segments (e.g. single words).
        """
        segments = self._split_claim_into_segments(claim)
        out: List[Dict[str, Any]] = []
        uncertainty_terms = {"less", "uncertain", "unclear", "inconclusive", "mixed", "limited", "insufficient"}
        assertive_claim = bool(
            re.search(r"\b(helps?|prevents?|reduces?|increases?|causes?|proves?|protects?)\b", claim, re.IGNORECASE)
        )
        for seg in segments:
            seg_words = set(re.findall(r"\b\w+\b", seg.lower()))
            seg_tokens = {w.lower() for w in re.findall(r"\b[\w']+\b", seg)}
            seg_neg = any(t in seg_tokens for t in {"no", "not", "never", "without", "lack", "lacks", "lacking"})
            best = None
            best_score = 0.0
            best_neg = False
            best_idx = -1
            best_anchor_ok = False
            for idx, ev in enumerate(evidence[:8]):
                stmt = (ev.get("statement") or ev.get("text") or "").strip()
                if not stmt:
                    continue
                if not self._segment_topic_guard_ok(seg, stmt):
                    continue
                stmt_words = set(re.findall(r"\b\w+\b", stmt.lower()))
                overlap = len(seg_words & stmt_words) / max(1, len(seg_words))
                anchor_eval = evaluate_anchor_match(seg, stmt)
                anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                anchor_ok = bool(anchor_eval.get("anchor_ok", False))
                rel = ev.get("final_score")
                if rel is None:
                    rel = ev.get("score", 0.0)
                try:
                    rel_f = float(rel or 0.0)
                except Exception:
                    rel_f = 0.0
                stmt_l = stmt.lower()
                stmt_tokens = {w.lower() for w in re.findall(r"\b[\w']+\b", stmt)}
                stmt_neg = any(
                    t in stmt_tokens
                    for t in {
                        "no",
                        "not",
                        "never",
                        "without",
                        "lack",
                        "lacks",
                        "lacking",
                        "myth",
                        "debunked",
                        "doesn't",
                        "isn't",
                        "cannot",
                        "can't",
                    }
                )
                uncertainty_penalty = 0.0
                if assertive_claim and any(t in stmt_l for t in uncertainty_terms):
                    uncertainty_penalty = 0.18
                reporting_penalty = 0.0
                if self._is_reporting_statement(stmt):
                    reporting_penalty = 0.24
                score = (
                    (0.50 * max(0.0, min(1.0, rel_f)))
                    + (0.25 * max(0.0, min(1.0, overlap)))
                    + (0.25 * max(0.0, min(1.0, anchor_overlap)))
                    - uncertainty_penalty
                    - reporting_penalty
                )
                if not anchor_ok:
                    score *= 0.35
                elif overlap < 0.20:
                    score *= 0.45
                if score > best_score:
                    best_score = score
                    best = ev
                    best_neg = stmt_neg
                    best_idx = idx
                    best_anchor_ok = anchor_ok

            if best and best_anchor_ok and best_score >= 0.55 and (best_neg == seg_neg):
                status = "VALID"
            elif best and best_anchor_ok and best_score >= 0.55 and (best_neg != seg_neg):
                status = "INVALID"
            elif best and best_anchor_ok and best_score >= 0.28 and (best_neg == seg_neg):
                status = "PARTIALLY_VALID"
            elif best and best_anchor_ok and best_score >= 0.28 and (best_neg != seg_neg):
                status = "PARTIALLY_INVALID"
            else:
                status = "UNKNOWN"

            out.append(
                {
                    "claim_segment": seg,
                    "status": status,
                    "supporting_fact": (best.get("statement") if best and status != "UNKNOWN" else "") or "",
                    "source_url": (best.get("source_url") if best and status != "UNKNOWN" else "") or "",
                    "evidence_used_ids": ([best_idx] if best and status != "UNKNOWN" and best_idx >= 0 else []),
                }
            )
        return out

    def _calculate_truthfulness_from_evidence(self, claim: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Evidence-driven truthfulness score based on:
        - segment-level best evidence support
        - semantic/kg relevance (not rank final_score)
        - lexical overlap between segment and evidence statement
        - source credibility
        - contradiction penalty using simple negation detection
        - diversity adjustment to avoid single-source inflation
        """
        if not evidence:
            return 0.0

        segments = self._split_claim_into_segments(claim)
        if not segments:
            segments = [claim]

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
            "could",
            "would",
            "should",
            "can",
        }
        neg_terms = {"no", "not", "never", "none", "without", "lack", "lacks", "lacking"}
        uncertainty_terms = {"uncertain", "unclear", "inconclusive", "mixed", "limited", "insufficient"}
        claim_assertive = bool(
            re.search(r"\b(helps?|prevents?|reduces?|increases?|causes?|proves?|protects?)\b", claim, re.IGNORECASE)
        )

        # Diversity adjustment based on unique domains in top evidence
        domains = set()
        for ev in evidence[:8]:
            src = ev.get("source_url") or ev.get("source") or ""
            if src:
                try:
                    domain = src.split("/")[2].lower()
                    if domain.startswith("www."):
                        domain = domain[4:]
                    domains.add(domain)
                except Exception:
                    pass
        diversity = len(domains) / max(1, len(evidence[:8]))
        diversity = max(0.4, min(1.0, diversity))

        def _semantic_strength(ev: Dict[str, Any]) -> float:
            sem_raw = ev.get("semantic_score")
            if sem_raw is None:
                sem_raw = ev.get("sem_score")
            if sem_raw is None:
                sem_raw = ev.get("final_score")
            kg_raw = ev.get("kg_score")
            try:
                sem_score = float(sem_raw or 0.0)
            except Exception:
                sem_score = 0.0
            try:
                kg_score = float(kg_raw or 0.0)
            except Exception:
                kg_score = 0.0
            sem_score = max(0.0, min(1.0, sem_score))
            kg_score = max(0.0, min(1.0, kg_score))
            # Entailment-style relevance signal (semantic first, KG as support).
            return max(0.0, min(1.0, (0.7 * sem_score) + (0.3 * kg_score)))

        def _segment_score(segment: str) -> float:
            seg_words = [w for w in re.findall(r"\b\w+\b", (segment or "").lower()) if w not in stop]
            seg_set = set(seg_words)
            if not seg_set:
                return 0.0
            seg_has_neg = any(t in seg_set for t in neg_terms)

            support_scores: List[float] = []
            contradiction_scores: List[float] = []
            best_src = ""
            for ev in evidence[:8]:
                stmt = (ev.get("statement") or ev.get("text") or "").strip()
                if not stmt:
                    continue
                if not self._segment_topic_guard_ok(segment, stmt):
                    continue

                rel = _semantic_strength(ev)

                credibility = ev.get("credibility")
                try:
                    cred = float(credibility if credibility is not None else 0.5)
                except Exception:
                    cred = 0.5
                cred = max(0.0, min(1.0, cred))

                stmt_words = [w for w in re.findall(r"\b\w+\b", stmt.lower()) if w not in stop]
                stmt_set = set(stmt_words)
                overlap = len(seg_set & stmt_set)
                overlap_ratio = overlap / max(1, len(seg_set))
                anchor_eval = evaluate_anchor_match(segment, stmt)
                anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                anchor_ok = bool(anchor_eval.get("anchor_ok", False))
                if not anchor_ok:
                    continue
                if overlap_ratio < 0.12 and rel < 0.75:
                    continue

                stance = str(ev.get("stance", "neutral") or "neutral").lower()
                uncertainty_penalty = 0.0
                if claim_assertive and any(t in stmt_set for t in uncertainty_terms):
                    uncertainty_penalty = 0.22
                reporting_penalty = 0.0
                if self._is_reporting_statement(stmt):
                    reporting_penalty = 0.24

                support = (0.70 * rel) + (0.15 * anchor_overlap) + (0.10 * overlap_ratio) + (0.05 * cred)
                if overlap_ratio < 0.20:
                    support *= 0.60
                support = max(0.0, min(1.0, support - uncertainty_penalty - reporting_penalty))

                polarity = self._segment_polarity(segment, stmt, stance=stance)
                if polarity == "contradicts":
                    contradiction_scores.append(support)
                    continue
                # Top-N supportive aggregation only; neutral/irrelevant lines do not dilute.
                if polarity == "entails" or support >= 0.40:
                    support_scores.append(support)
                    if support >= 0.60 and not best_src:
                        best_src = ev.get("source_url") or ev.get("source") or ""

            top_support = sorted(support_scores, reverse=True)[:3]
            avg_support = (sum(top_support) / len(top_support)) if top_support else 0.0
            contradiction_penalty = (max(contradiction_scores) if contradiction_scores else 0.0) * 0.55
            neg_alignment_penalty = 0.0
            if seg_has_neg and top_support and avg_support < 0.45:
                neg_alignment_penalty = 0.08
            best = max(0.0, min(1.0, avg_support - contradiction_penalty - neg_alignment_penalty))
            if best_src:
                logger.info(
                    "[VerdictGenerator] Segment truthfulness best=%.3f src=%s segment=%s",
                    best,
                    best_src,
                    segment[:60],
                )
            return best

        segment_scores = [_segment_score(seg) for seg in segments]
        if not segment_scores:
            return 0.0

        avg_support = sum(segment_scores) / len(segment_scores)
        # Keep diversity as a mild reliability adjustment, not a hard multiplier.
        truthfulness = avg_support * (0.85 + (0.15 * diversity))
        ceiling = 0.85 + (0.10 * diversity)
        truthfulness = min(truthfulness, ceiling)
        return round(truthfulness * 100.0, 1)

    def _calculate_confidence(self, evidence: List[Dict[str, Any]], claim_breakdown: List[Dict[str, Any]]) -> float:
        """
        Confidence means certainty in the estimated truthfulness score, not positivity.
        High confidence is valid even when truthfulness is low if evidence is strong/consistent.
        """
        scores = []
        for ev in evidence[:5]:
            s = ev.get("final_score")
            if s is None:
                s = ev.get("score", 0.0)
            try:
                scores.append(float(s or 0.0))
            except Exception:
                continue

        avg_score = sum(scores) / max(1, len(scores))

        unknown_ratio = 0.0
        certainty_ratio = 0.0
        conflict_ratio = 0.0
        if claim_breakdown:
            statuses = [s.get("status", "UNKNOWN").upper() for s in claim_breakdown]
            unknown_ratio = sum(1 for s in statuses if s == "UNKNOWN") / max(1, len(statuses))
            certainty_weight = {
                "VALID": 1.0,
                "INVALID": 1.0,
                "PARTIALLY_VALID": 0.7,
                "PARTIALLY_INVALID": 0.7,
                "UNKNOWN": 0.0,
            }
            certainty_ratio = sum(certainty_weight.get(s, 0.0) for s in statuses) / max(1, len(statuses))
            has_support = any(s in {"VALID", "PARTIALLY_VALID"} for s in statuses)
            has_contra = any(s in {"INVALID", "PARTIALLY_INVALID"} for s in statuses)
            conflict_ratio = 0.25 if (has_support and has_contra) else 0.0

        cred_scores = []
        for ev in evidence[:5]:
            c = ev.get("credibility")
            try:
                cred_scores.append(float(c if c is not None else 0.5))
            except Exception:
                cred_scores.append(0.5)
        avg_cred = sum(cred_scores) / max(1, len(cred_scores))

        count_factor = min(len(evidence), 5) / 5.0
        low_count_penalty = 0.0
        if len(evidence) < 3:
            low_count_penalty = 0.20 * ((3 - len(evidence)) / 2.0)

        domains = set()
        for ev in evidence[:5]:
            src = ev.get("source_url") or ev.get("source") or ""
            try:
                domain = src.split("/")[2].lower()
                if domain.startswith("www."):
                    domain = domain[4:]
                if domain:
                    domains.add(domain)
            except Exception:
                continue
        diversity = min(1.0, len(domains) / max(1, min(len(evidence), 5)))
        evidence_strength = (0.55 * avg_score) + (0.30 * avg_cred) + (0.15 * count_factor)
        certainty = (0.75 * certainty_ratio) + (0.25 * diversity)
        supported_ratio = 0.0
        if claim_breakdown:
            supported_ratio = sum(
                1
                for s in claim_breakdown
                if (s.get("status") or "").upper() in {"VALID", "INVALID", "PARTIALLY_VALID", "PARTIALLY_INVALID"}
            ) / max(1, len(claim_breakdown))
        coverage_penalty = 0.15 if supported_ratio < 0.50 else 0.0

        logger.debug(
            "[VerdictGenerator] Confidence inputs: avg_score=%.3f avg_cred=%.3f "
            "certainty_ratio=%.2f unknown_ratio=%.2f conflict_ratio=%.2f diversity=%.2f "
            "supported_ratio=%.2f low_count_penalty=%.2f coverage_penalty=%.2f evidence_n=%d",
            avg_score,
            avg_cred,
            certainty_ratio,
            unknown_ratio,
            conflict_ratio,
            diversity,
            supported_ratio,
            low_count_penalty,
            coverage_penalty,
            len(evidence),
        )

        confidence = (
            0.05
            + (0.55 * evidence_strength)
            + (0.35 * certainty)
            - (0.30 * unknown_ratio)
            - conflict_ratio
            - low_count_penalty
            - coverage_penalty
        )
        return max(0.05, min(0.98, confidence))

    def _build_default_evidence_map(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build default evidence map when LLM doesn't provide one."""
        evidence_map = []
        for i, ev in enumerate(evidence):
            score = float(ev.get("final_score", ev.get("score", 0.0)) or 0.0)
            anchor_score = float(ev.get("anchor_match_score", 0.0) or 0.0)
            if score >= 0.65 and anchor_score >= 0.20:
                relevance = "SUPPORTS"
            elif score >= 0.45 and anchor_score >= 0.20:
                relevance = "PARTIAL"
            else:
                relevance = "NEUTRAL"
            evidence_map.append(
                {
                    "evidence_id": i,
                    "statement": ev.get("statement", ""),
                    "relevance": relevance,
                    "relevance_score": score,
                    "source_url": ev.get("source_url", ""),
                    "anchor_match_score": anchor_score,
                }
            )
        return evidence_map

    def _unverifiable_result(self, claim: str, reason: str) -> Dict[str, Any]:
        """Return a default UNVERIFIABLE result."""
        return {
            "verdict": Verdict.UNVERIFIABLE.value,
            "confidence": 0.0,
            "truthfulness_percent": 0,
            "rationale": reason,
            "claim_breakdown": [],
            "evidence_map": [],
            "key_findings": [],
            "claim": claim,
            "evidence_count": 0,
        }

    # ================================================================
    # CLAIM-SEGMENT RETRIEVAL METHODS
    # ================================================================

    def _split_claim_into_segments(self, claim: str) -> List[str]:
        """
        Split claim into logical segments for independent retrieval.
        Uses shared deterministic segmentation so verdict and adaptive trust
        operate on the exact same segment boundaries.
        """
        segments = split_claim_into_segments(claim)
        filtered = [s for s in segments if self._is_meaningful_segment(s)]
        if not filtered:
            cleaned = re.sub(r"\s+", " ", (claim or "")).strip(" ,.")
            filtered = [cleaned] if cleaned else []
        logger.debug(f"[VerdictGenerator] Split claim into {len(filtered)} segments")
        return filtered

    async def _retrieve_segment_evidence_for_segments(
        self, segments: List[str], top_k: int = 2, max_segments: int = 3
    ) -> List[Dict[str, Any]]:
        """Query VDB for specific segments independently and dedupe by statement."""
        segments = [s for s in (segments or []) if s][:max_segments]
        all_segment_evidence: List[Dict[str, Any]] = []
        seen_statements: set = set()

        for segment in segments:
            try:
                topics, _ = await self.topic_classifier.classify(segment, [], None)
                topic_filter = topics or None
                if not topics:
                    logger.warning(
                        f"[VerdictGenerator] No topics for segment '{segment[:30]}...'; "
                        "running VDB retrieval without topic filter"
                    )
                results = await self.vdb_retriever.search(segment, top_k=top_k, topics=topic_filter)
                for result in results:
                    stmt = result.get("statement", "")
                    # Deduplicate by statement
                    if stmt and stmt not in seen_statements:
                        seen_statements.add(stmt)
                        # Mark as segment-retrieved for debugging
                        result["_segment_query"] = segment[:50]
                        all_segment_evidence.append(result)
            except Exception as e:
                logger.warning(f"[VerdictGenerator] Segment retrieval failed for '{segment[:30]}...': {e}")

        logger.info(
            f"[VerdictGenerator] Segment retrieval: {len(segments)} segments -> "
            f"{len(all_segment_evidence)} unique facts"
        )
        return all_segment_evidence

    async def _retrieve_segment_evidence(self, claim: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Query VDB for each claim segment independently.

        OPTIMIZATION: Reduced top_k from 3 to 2 per segment.
        Uses max 3 segments to limit VDB queries.
        """
        segments = self._split_claim_into_segments(claim)
        return await self._retrieve_segment_evidence_for_segments(segments, top_k=top_k, max_segments=3)

    @staticmethod
    def _normalize_statement_key(statement: str) -> str:
        return re.sub(r"\s+", " ", (statement or "").strip().lower())

    @staticmethod
    def _deterministic_evidence_sort_key(ev: Dict[str, Any]) -> tuple[float, str, str]:
        score = ev.get("final_score")
        if score is None:
            score = ev.get("score")
        if score is None:
            score = ev.get("sem_score")
        if score is None:
            score = ev.get("semantic_score")
        try:
            score_f = float(score or 0.0)
        except Exception:
            score_f = 0.0
        source = str(ev.get("source_url") or ev.get("source") or "").strip().lower()
        stmt = re.sub(r"\s+", " ", str(ev.get("statement") or ev.get("text") or "").strip().lower())
        return (-score_f, source, stmt)

    def _merge_evidence(
        self,
        ranked_evidence: List[Dict[str, Any]],
        segment_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge ranked evidence with segment-specific evidence.

        Ranked evidence comes first (already ranked by relevance).
        Segment evidence is appended if not already present.
        Filters out items with None/empty statements.
        """
        # Filter ranked evidence - keep only items with valid statements
        merged: List[Dict[str, Any]] = [ev for ev in ranked_evidence if ev.get("statement") or ev.get("text")]
        seen_fingerprints: set[str] = set()
        for ev in merged:
            stmt = self._normalize_statement_key(str(ev.get("statement") or ev.get("text") or ""))
            src = str(ev.get("source_url") or ev.get("source") or "").strip().lower()
            if stmt:
                seen_fingerprints.add(sha1(f"{stmt}|{src}".encode("utf-8")).hexdigest())

        for seg_ev in segment_evidence:
            stmt = seg_ev.get("statement") or seg_ev.get("text", "")
            stmt_key = self._normalize_statement_key(str(stmt))
            src = str(seg_ev.get("source_url") or seg_ev.get("source") or "").strip().lower()
            fingerprint = sha1(f"{stmt_key}|{src}".encode("utf-8")).hexdigest() if stmt_key else ""
            if stmt and stmt_key and fingerprint and fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                merged.append(seg_ev)

        merged.sort(key=self._deterministic_evidence_sort_key)
        return merged

    def _log_low_confidence_unverifiable_reason(
        self,
        claim: str,
        ranked_evidence: List[Dict[str, Any]],
        final_evidence: List[Dict[str, Any]],
        verdict_result: Dict[str, Any],
    ) -> None:
        verdict = str(verdict_result.get("verdict", "") or "").upper()
        confidence = float(verdict_result.get("confidence", 0.0) or 0.0)
        if verdict != Verdict.UNVERIFIABLE.value or confidence > 0.10:
            return

        claim_breakdown = verdict_result.get("claim_breakdown", []) or []
        unknown_count = sum(1 for item in claim_breakdown if str(item.get("status", "")).upper() == "UNKNOWN")
        reason = "coverage_remained_unknown"
        if not ranked_evidence:
            reason = "retrieval_empty"
        elif not final_evidence:
            reason = "zero_facts_extracted"
        elif self._policy_says_insufficient(claim, final_evidence):
            reason = "gating_insufficient_evidence"
        elif unknown_count > 0:
            reason = "coverage_remained_unknown"

        logger.warning(
            "[VerdictGenerator][LowConfidence] UNVERIFIABLE<=0.10 reason=%s ranked=%d final=%d unknown_segments=%d",
            reason,
            len(ranked_evidence),
            len(final_evidence),
            unknown_count,
        )

    def _get_unknown_segments(self, verdict_result: Dict[str, Any]) -> List[str]:
        """Extract segments marked as UNKNOWN from the verdict result."""
        unknown_segments = []
        claim_breakdown = verdict_result.get("claim_breakdown", [])

        for segment in claim_breakdown:
            if segment.get("status") == "UNKNOWN":
                claim_segment = segment.get("claim_segment", "")
                if claim_segment:
                    unknown_segments.append(claim_segment)

        return unknown_segments

    async def _fetch_web_evidence_for_unknown_segments(
        self,
        unknown_segments: List[str],
        max_queries_per_segment: int = 2,
        max_urls_per_query: int = 3,
    ) -> List[Dict[str, Any]]:
        """Fetch web evidence for UNKNOWN claim segments."""
        all_web_evidence = []

        for segment in unknown_segments:
            try:
                logger.info(f"[VerdictGenerator] Searching web for UNKNOWN segment: '{segment[:50]}...'")

                # Generate deterministic + site-specific queries for this segment
                queries = await self.trusted_search.generate_search_queries(
                    post_text=segment,
                    failed_entities=[],
                    max_queries=max_queries_per_segment,
                    subclaims=[segment],
                    entities=[],
                )
                if not queries:
                    logger.warning(f"[VerdictGenerator] No search queries generated for segment: {segment[:30]}...")
                    continue

                for query in queries[:max_queries_per_segment]:
                    logger.info(f"[VerdictGenerator] Using search query: '{query}'")

                    # Perform the search
                    search_results = await self.trusted_search.search(query, max_results=5)
                    if not search_results:
                        logger.warning(f"[VerdictGenerator] No search results for query: {query}")
                        continue

                    # Extract facts from search results
                    for result in search_results[:max_urls_per_query]:
                        url = result.get("url", "")
                        if not url:
                            continue

                        try:
                            logger.debug(f"[VerdictGenerator] Scraping and extracting facts from: {url}")

                            # Scrape the URL to get content
                            import aiohttp

                            async with aiohttp.ClientSession() as session:
                                scraped_page = await self.scraper.scrape_one(session, url)

                            if not scraped_page.get("content"):
                                logger.warning(f"[VerdictGenerator] No content scraped from {url}")
                                continue

                            # Extract facts from the scraped content
                            facts = await self.fact_extractor.extract([scraped_page])

                            for fact in facts:
                                stmt = fact.get("statement", "") or ""
                                conf = float(fact.get("confidence", 0.5) or 0.5)
                                score = self._quick_web_score(segment, stmt, conf)
                                evidence_item = {
                                    "statement": stmt,
                                    "source_url": url,
                                    "final_score": score,
                                    "extraction_confidence": conf,
                                    "credibility": 0.7,  # Default credibility for web-extracted facts
                                    "_web_search": True,  # Mark as web-sourced
                                    "_original_query": query,
                                }
                                all_web_evidence.append(evidence_item)

                        except Exception as e:
                            logger.warning(f"[VerdictGenerator] Failed to extract facts from {url}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"[VerdictGenerator] Failed to fetch web evidence for segment '{segment[:30]}...': {e}")
                continue

        logger.info(
            f"[VerdictGenerator] Retrieved {len(all_web_evidence)} web evidence items "
            f"for {len(unknown_segments)} UNKNOWN segments"
        )
        return all_web_evidence
