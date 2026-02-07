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

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from app.constants.config import LLM_TEMPERATURE_VERDICT
from app.core.logger import get_logger
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.scraper import Scraper
from app.services.corrective.trusted_search import TrustedSearch
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority
from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy
from app.services.retrieval.metadata_enricher import TopicClassifier
from app.services.vdb.vdb_retrieval import VDBRetrieval

logger = get_logger(__name__)


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
        self.vdb_retriever = vdb_retriever or VDBRetrieval()
        self.trusted_search = TrustedSearch()
        self.fact_extractor = FactExtractor()
        self.scraper = Scraper()
        self.trust_policy = AdaptiveTrustPolicy()
        self.topic_classifier = TopicClassifier()
        self.MAX_WEB_ROUNDS_PRE_VERDICT = 2
        self.WEB_SEGMENTS_LIMIT = 3
        self.MAX_UNKNOWN_ROUNDS_POST_VERDICT = 2

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
        metrics = self.trust_policy.compute_adaptive_trust(claim, adapted, top_k=min(10, len(adapted)))
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
            ranked_evidence = web_boost

        # Step 1) Start with VDB/KG evidence + (optional) segment retrieval
        segment_evidence: List[Dict[str, Any]] = []
        if (not used_web_search and not cache_sufficient) and (
            self._needs_web_boost(ranked_evidence[: min(len(ranked_evidence), 6)], claim=claim)
            or self._policy_says_insufficient(claim, ranked_evidence[: min(len(ranked_evidence), 10)])
        ):
            segment_evidence = await self._retrieve_segment_evidence(claim, top_k=2)

        enriched_evidence = self._merge_evidence(ranked_evidence, segment_evidence)

        if segment_evidence:
            logger.info(
                f"[VerdictGenerator] Evidence: {len(ranked_evidence)} ranked + "
                f"{len(segment_evidence)} segment-specific = {len(enriched_evidence)} total"
            )
        else:
            logger.info(f"[VerdictGenerator] Using {len(ranked_evidence)} ranked evidence (segment retrieval skipped)")

        # Step 2) PRE-VERDICT web-boost loop driven by *sufficiency*
        pre_evidence = enriched_evidence[: min(len(enriched_evidence), max(top_k, 8), 12)]
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
                pre_evidence = (pre_evidence + web_boost)[: min(len(pre_evidence + web_boost), 18)]
        top_evidence = pre_evidence[: min(len(pre_evidence), top_k, 12)]

        # Format evidence for prompt
        evidence_text = self._format_evidence_for_prompt(top_evidence)

        # Build RAG prompt
        prompt = VERDICT_GENERATION_PROMPT.format(claim=claim, evidence_text=evidence_text)

        try:
            # HIGH priority - verdict generation is critical
            # Use low temperature for consistent claim segmentation/breakdown
            result = await self.llm_service.ainvoke(
                prompt, response_format="json", priority=LLMPriority.HIGH, temperature=LLM_TEMPERATURE_VERDICT
            )

            # Validate and parse result
            verdict_result = self._parse_verdict_result(result, claim, top_evidence)

            logger.info(
                f"[VerdictGenerator] Generated verdict: {verdict_result['verdict']} "
                f"(confidence: {verdict_result['confidence']:.2f})"
            )

            # If UNKNOWN segments remain, iterate a couple times (bounded) to avoid UNKNOWN
            if not used_web_search and not cache_sufficient:
                for _ in range(self.MAX_UNKNOWN_ROUNDS_POST_VERDICT):
                    unknown_segments = self._get_unknown_segments(verdict_result)
                    if not unknown_segments:
                        break
                    logger.info(
                        f"[VerdictGenerator] Found {len(unknown_segments)} UNKNOWN segments, fetching web evidence..."
                    )
                    web_evidence = await self._fetch_web_evidence_for_unknown_segments(unknown_segments)
                    if web_evidence:
                        logger.info(f"[VerdictGenerator] Retrieved {len(web_evidence)} additional facts from web")
                        # Merge web evidence with existing evidence
                        enriched_evidence = top_evidence + web_evidence
                        # Re-run verdict generation with additional evidence
                        enriched_evidence_text = self._format_evidence_for_prompt(enriched_evidence)
                        enriched_prompt = VERDICT_GENERATION_PROMPT.format(
                            claim=claim, evidence_text=enriched_evidence_text
                        )

                        try:
                            enriched_result = await self.llm_service.ainvoke(
                                enriched_prompt,
                                response_format="json",
                                priority=LLMPriority.HIGH,
                                temperature=LLM_TEMPERATURE_VERDICT,
                            )
                            verdict_result = self._parse_verdict_result(enriched_result, claim, enriched_evidence)
                            logger.info(
                                f"[VerdictGenerator] Re-generated verdict with web evidence: "
                                f"{verdict_result['verdict']} "
                                f"(confidence: {verdict_result['confidence']:.2f})"
                            )
                        except Exception as e:
                            logger.warning(f"[VerdictGenerator] Failed to re-generate verdict with web evidence: {e}")
                            break

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
    ) -> Dict[str, Any]:
        """Parse and validate LLM result, with fallback handling."""
        # Handle LLM error indicators
        if "_llm_error" in llm_result:
            return self._unverifiable_result(claim, llm_result.get("raw_text", "LLM error"))

        # Extract verdict with validation
        verdict_str = llm_result.get("verdict", "UNVERIFIABLE").upper()
        if verdict_str not in [v.value for v in Verdict]:
            verdict_str = "UNVERIFIABLE"

        # Extract confidence (will be re-scored from evidence if possible)
        confidence = llm_result.get("confidence", 0.5)
        confidence = max(0.0, min(1.0, float(confidence)))

        # Extract rationale
        rationale = llm_result.get("rationale", "No rationale provided")

        # Extract evidence map or build from evidence
        evidence_map = llm_result.get("evidence_map", [])
        if not evidence_map:
            evidence_map = self._build_default_evidence_map(evidence)

        # Extract key findings
        key_findings = llm_result.get("key_findings", [])

        # Extract claim breakdown for client display
        claim_breakdown = llm_result.get("claim_breakdown", [])
        if self._should_rebuild_claim_breakdown(claim_breakdown):
            claim_breakdown = self._build_deterministic_claim_breakdown(claim, evidence)

        # Guardrail: prevent hallucinated support (source_url / supporting_fact must map to provided evidence)
        ev_urls = set((e.get("source_url") or e.get("source") or "") for e in evidence)
        ev_text = " ".join((e.get("statement") or e.get("text") or "") for e in evidence).lower()

        def _overlap_ok(fact: str, ev_text: str) -> bool:
            if not fact:
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
            fw = [w for w in re.findall(r"\b\w+\b", fact.lower()) if w not in stop]
            if len(fw) < 3:
                return True  # don't over-penalize short facts
            hits = sum(1 for w in set(fw) if w in ev_text)
            return hits >= 2

        for seg in claim_breakdown:
            status = (seg.get("status") or "UNKNOWN").upper()
            if status == "UNKNOWN":
                continue
            src = seg.get("source_url") or ""
            fact = (seg.get("supporting_fact") or "").strip().lower()
            if (src and src not in ev_urls) or (fact and not _overlap_ok(fact, ev_text)):
                seg["status"] = "UNKNOWN"
                seg["supporting_fact"] = ""
                seg["source_url"] = ""

        # Calculate truthfulness from evidence (deterministic, claim-aware)
        truthfulness_percent = self._calculate_truthfulness_from_evidence(claim, evidence)

        # Re-score confidence from evidence + breakdown when possible
        if evidence:
            confidence = self._calculate_confidence(evidence, claim_breakdown)

        return {
            "verdict": verdict_str,
            "confidence": confidence,
            "truthfulness_percent": truthfulness_percent,
            "rationale": rationale,
            "claim_breakdown": claim_breakdown,
            "evidence_map": evidence_map,
            "key_findings": key_findings,
            "claim": claim,
            "evidence_count": len(evidence),
        }

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

    def _should_rebuild_claim_breakdown(self, claim_breakdown: List[Dict[str, Any]]) -> bool:
        if not claim_breakdown:
            return True
        low_quality = 0
        for item in claim_breakdown:
            seg = (item.get("claim_segment") or "").strip()
            if not self._is_meaningful_segment(seg):
                low_quality += 1
        return low_quality > 0

    def _build_deterministic_claim_breakdown(self, claim: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build a deterministic claim breakdown with meaningful segments.
        Used when LLM returns fragmentary segments (e.g. single words).
        """
        segments = self._split_claim_into_segments(claim)
        out: List[Dict[str, Any]] = []
        for seg in segments:
            seg_words = set(re.findall(r"\b\w+\b", seg.lower()))
            best = None
            best_score = 0.0
            for ev in evidence[:8]:
                stmt = (ev.get("statement") or ev.get("text") or "").strip()
                if not stmt:
                    continue
                stmt_words = set(re.findall(r"\b\w+\b", stmt.lower()))
                overlap = len(seg_words & stmt_words) / max(1, len(seg_words))
                rel = ev.get("final_score")
                if rel is None:
                    rel = ev.get("score", 0.0)
                try:
                    rel_f = float(rel or 0.0)
                except Exception:
                    rel_f = 0.0
                score = (0.6 * max(0.0, min(1.0, rel_f))) + (0.4 * max(0.0, min(1.0, overlap)))
                if score > best_score:
                    best_score = score
                    best = ev

            if best and best_score >= 0.60:
                status = "VALID"
            elif best and best_score >= 0.38:
                status = "PARTIALLY_VALID"
            else:
                status = "UNKNOWN"

            out.append(
                {
                    "claim_segment": seg,
                    "status": status,
                    "supporting_fact": (best.get("statement") if best and status != "UNKNOWN" else "") or "",
                    "source_url": (best.get("source_url") if best and status != "UNKNOWN" else "") or "",
                }
            )
        return out

    def _calculate_truthfulness_from_evidence(self, claim: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Evidence-driven truthfulness score based on:
        - segment-level best evidence support
        - semantic relevance (final_score / sem_score)
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
        diversity = max(0.5, min(1.0, diversity))

        def _segment_score(segment: str) -> float:
            seg_words = [w for w in re.findall(r"\b\w+\b", (segment or "").lower()) if w not in stop]
            seg_set = set(seg_words)
            if not seg_set:
                return 0.0
            seg_has_neg = any(t in seg_set for t in neg_terms)

            best = 0.0
            best_src = ""
            for ev in evidence[:8]:
                stmt = (ev.get("statement") or ev.get("text") or "").strip()
                if not stmt:
                    continue

                s = ev.get("final_score")
                if s is None:
                    s = ev.get("sem_score", ev.get("score", 0.0))
                try:
                    rel = float(s or 0.0)
                except Exception:
                    rel = 0.0
                rel = max(0.0, min(1.0, rel))

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

                stmt_has_neg = any(t in stmt_set for t in neg_terms)
                contradiction = 1.0 if (stmt_has_neg and not seg_has_neg and overlap_ratio >= 0.25) else 0.0

                support = (0.45 * rel) + (0.35 * overlap_ratio) + (0.20 * cred)
                net = support - (0.60 * contradiction)
                net = max(0.0, min(1.0, net))

                if net > best:
                    best = net
                    best_src = ev.get("source_url") or ev.get("source") or ""

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
        truthfulness = avg_support * diversity
        return round(truthfulness * 100.0, 1)

    def _calculate_confidence(self, evidence: List[Dict[str, Any]], claim_breakdown: List[Dict[str, Any]]) -> float:
        """
        Heuristic confidence score derived from evidence quality and breakdown certainty.
        This avoids static LLM confidence outputs across different claims.
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
        if claim_breakdown:
            statuses = [s.get("status", "UNKNOWN").upper() for s in claim_breakdown]
            unknown_ratio = sum(1 for s in statuses if s == "UNKNOWN") / max(1, len(statuses))

        logger.debug(
            "[VerdictGenerator] Confidence inputs: avg_score=%.3f unknown_ratio=%.2f evidence_n=%d",
            avg_score,
            unknown_ratio,
            len(evidence),
        )

        # Base + evidence quality - uncertainty penalty
        confidence = 0.2 + (0.7 * avg_score) - (0.3 * unknown_ratio)
        return max(0.05, min(0.98, confidence))

    def _build_default_evidence_map(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build default evidence map when LLM doesn't provide one."""
        evidence_map = []
        for i, ev in enumerate(evidence):
            evidence_map.append(
                {
                    "evidence_id": i,
                    "statement": ev.get("statement", ""),
                    "relevance": "NEUTRAL",  # Default when unknown
                    "relevance_score": ev.get("final_score", ev.get("score", 0)),
                    "source_url": ev.get("source_url", ""),
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

        Uses sentence boundaries and minimal connectors to split.
        Each segment should be a complete, verifiable statement.
        """

        def _clean_spaces(t: str) -> str:
            return re.sub(r"\s+", " ", t).strip(" ,.")

        def _expand_enumeration(sentence: str) -> List[str]:
            # Build meaningful segments for patterns like:
            # "A diet rich in fruits, vegetables, and low in saturated fats helps prevent ..."
            pred = re.search(
                (
                    r"\b(helps?|prevents?|reduces?|increases?|causes?|improves?|worsens?|protects?|"
                    r"associated with|linked to|leads to)\b"
                ),
                sentence,
                flags=re.IGNORECASE,
            )
            if not pred:
                return []
            head = _clean_spaces(sentence[: pred.start()])
            tail = _clean_spaces(sentence[pred.start() :])
            if ("," not in head) and (" and " not in head.lower()):
                return []

            normalized = re.sub(r"\s*,\s*and\s+", ", ", head, flags=re.IGNORECASE)
            normalized = re.sub(r"\s+and\s+", ", ", normalized, flags=re.IGNORECASE)
            items = [_clean_spaces(x) for x in normalized.split(",") if _clean_spaces(x)]
            if len(items) < 2:
                return []

            first = items[0]
            q = re.search(r"\b(rich in|low in|high in|with|without|deficient in)\b", first, flags=re.IGNORECASE)
            subject_root = _clean_spaces(first[: q.start()]) if q else _clean_spaces(" ".join(first.split()[:2]))
            qualifier_prefix = (first[: q.end()].strip() + " ") if q else ""

            out: List[str] = []
            for idx, item in enumerate(items):
                phrase = item
                if idx > 0 and len(item.split()) <= 4:
                    if qualifier_prefix:
                        phrase = qualifier_prefix + item
                    elif subject_root:
                        phrase = f"{subject_root} {item}"
                elif (
                    idx > 0
                    and subject_root
                    and re.match(r"^(rich in|low in|high in|with|without|deficient in)\b", item, flags=re.IGNORECASE)
                ):
                    phrase = f"{subject_root} {item}"

                seg = _clean_spaces(f"{phrase} {tail}")
                if self._is_meaningful_segment(seg):
                    out.append(seg)
            return out

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", claim) if s.strip()]
        segments: List[str] = []
        for sentence in sentences:
            expanded = _expand_enumeration(sentence)
            if expanded:
                segments.extend(expanded)
                continue
            parts = [p.strip() for p in re.split(r"\s*;\s*", sentence) if p.strip()]
            segments.extend(parts or [sentence])

        filtered: List[str] = []
        seen = set()
        for seg in segments:
            seg = _clean_spaces(seg)
            if not self._is_meaningful_segment(seg):
                continue
            key = seg.lower()
            if key not in seen:
                seen.add(key)
                filtered.append(seg)

        if not filtered:
            filtered = [_clean_spaces(claim)]

        logger.debug(f"[VerdictGenerator] Split claim into {len(filtered)} segments")
        return filtered

    async def _retrieve_segment_evidence(self, claim: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Query VDB for each claim segment independently.

        OPTIMIZATION: Reduced top_k from 3 to 2 per segment.
        Uses max 3 segments to limit VDB queries.
        """
        segments = self._split_claim_into_segments(claim)

        # OPTIMIZATION: Limit to max 3 segments to control VDB query count
        segments = segments[:3]

        all_segment_evidence: List[Dict[str, Any]] = []
        seen_statements: set = set()

        for segment in segments:
            try:
                topics, _ = await self.topic_classifier.classify(segment, [], None)
                if not topics:
                    logger.warning(
                        f"[VerdictGenerator] No topics for segment '{segment[:30]}...', skipping VDB retrieval"
                    )
                    continue
                results = await self.vdb_retriever.search(segment, top_k=top_k, topics=topics)
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
        seen_statements: set = {ev.get("statement") or ev.get("text", "") for ev in merged}

        for seg_ev in segment_evidence:
            stmt = seg_ev.get("statement") or seg_ev.get("text", "")
            if stmt and stmt not in seen_statements:
                seen_statements.add(stmt)
                merged.append(seg_ev)

        return merged

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

    async def _fetch_web_evidence_for_unknown_segments(self, unknown_segments: List[str]) -> List[Dict[str, Any]]:
        """Fetch web evidence for UNKNOWN claim segments."""
        all_web_evidence = []

        for segment in unknown_segments:
            try:
                logger.info(f"[VerdictGenerator] Searching web for UNKNOWN segment: '{segment[:50]}...'")

                # Generate deterministic + site-specific queries for this segment
                queries = await self.trusted_search.generate_search_queries(
                    post_text=segment,
                    failed_entities=[],
                    max_queries=2,
                    subclaims=[segment],
                    entities=[],
                )
                if not queries:
                    logger.warning(f"[VerdictGenerator] No search queries generated for segment: {segment[:30]}...")
                    continue

                for query in queries[:2]:
                    logger.info(f"[VerdictGenerator] Using search query: '{query}'")

                    # Perform the search
                    search_results = await self.trusted_search.search(query, max_results=5)
                    if not search_results:
                        logger.warning(f"[VerdictGenerator] No search results for query: {query}")
                        continue

                    # Extract facts from search results
                    for result in search_results[:3]:  # Limit to 3 URLs per segment
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
