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
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority
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
1. Extract EXACT text segments from the claim (copy-paste, no paraphrasing)
2. For each segment, search the evidence for supporting or contradicting facts
3. Determine each segment's status based on evidence strength and credibility
4. Calculate truthfulness_percent by comparing claim content against evidence

CRITICAL RULES FOR CLAIM SEGMENTS:
- claim_segment MUST be an EXACT substring copied directly from the original claim
- DO NOT paraphrase, summarize, or reword any part of the claim
- DO NOT add words that don't exist in the original claim
- Each segment should be a complete verifiable statement from the claim

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
            "claim_segment": "EXACT text copied from the claim - no changes allowed",
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

    async def generate_verdict(
        self,
        claim: str,
        ranked_evidence: List[Dict[str, Any]],
        top_k: int = 5,
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
            logger.warning("[VerdictGenerator] No evidence provided, returning UNVERIFIABLE")
            return self._unverifiable_result(claim, "No evidence retrieved")

        # ================================================================
        # CLAIM-SEGMENT RETRIEVAL: Only if ranked evidence is insufficient
        # OPTIMIZATION: Skip segment retrieval if we already have good evidence
        # ================================================================
        segment_evidence: List[Dict[str, Any]] = []

        # Only do segment retrieval if ranked evidence is sparse (<3 items)
        # This avoids unnecessary VDB queries when we already have enough evidence
        if len(ranked_evidence) < 3:
            segment_evidence = await self._retrieve_segment_evidence(claim, top_k=2)

        # Merge segment evidence with ranked evidence (dedup by statement)
        enriched_evidence = self._merge_evidence(ranked_evidence, segment_evidence)

        if segment_evidence:
            logger.info(
                f"[VerdictGenerator] Evidence: {len(ranked_evidence)} ranked + "
                f"{len(segment_evidence)} segment-specific = {len(enriched_evidence)} total"
            )
        else:
            logger.info(f"[VerdictGenerator] Using {len(ranked_evidence)} ranked evidence (segment retrieval skipped)")

        # OPTIMIZATION: Use exactly top_k evidence, capped at 6 to reduce prompt size
        # More evidence doesn't improve accuracy but increases latency and LLM cost
        evidence_limit = min(len(enriched_evidence), top_k, 6)
        top_evidence = enriched_evidence[:evidence_limit]

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

        # Extract confidence with bounds
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

        # Extract or calculate truthfulness percentage
        truthfulness_percent = llm_result.get("truthfulness_percent")
        if truthfulness_percent is None:
            # Calculate from claim_breakdown if not provided
            truthfulness_percent = self._calculate_truthfulness_percent(claim_breakdown)

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

    def _calculate_truthfulness_percent(self, claim_breakdown: List[Dict[str, Any]]) -> float:
        """
        Fallback calculation of truthfulness percentage from claim breakdown.
        Only used if LLM doesn't provide truthfulness_percent.

        Status weights:
        - VALID: 100%
        - PARTIALLY_VALID: 75%
        - UNKNOWN: 50%
        - PARTIALLY_INVALID: 25%
        - INVALID: 0%
        """
        if not claim_breakdown:
            return 50.0  # Default to 50% when no breakdown available

        status_weights = {
            "VALID": 100.0,
            "PARTIALLY_VALID": 75.0,
            "UNKNOWN": 50.0,
            "PARTIALLY_INVALID": 25.0,
            "INVALID": 0.0,
        }

        total = 0.0
        for segment in claim_breakdown:
            status = segment.get("status", "UNKNOWN").upper()
            total += status_weights.get(status, 50.0)

        return round(total / len(claim_breakdown), 1)

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

        Uses sentence boundaries and common connectors to split.
        Each segment should be a complete, verifiable statement.
        """
        # Split on sentence boundaries and common connectors
        delimiters = [
            r"\. ",  # Period followed by space
            r"; ",  # Semicolon
            r"\. However,",  # Common transition
            r", however,",
            r"\. Additionally,",
            r", and ",  # Compound statements
            r", but ",
        ]

        # Join delimiters into regex pattern
        pattern = "|".join(delimiters)

        # Split and clean
        raw_segments = re.split(pattern, claim, flags=re.IGNORECASE)

        # Clean up segments
        segments = []
        for seg in raw_segments:
            seg = seg.strip()
            # Skip very short segments (likely incomplete)
            if len(seg) > 20:
                segments.append(seg)

        # If splitting produced no good segments, use full claim
        if not segments:
            segments = [claim]

        logger.debug(f"[VerdictGenerator] Split claim into {len(segments)} segments")
        return segments

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
                results = await self.vdb_retriever.search(segment, top_k=top_k)
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


__all__ = ["VerdictGenerator", "Verdict"]
