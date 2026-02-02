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
1. Break down the claim into verifiable segments/statements
2. Analyze each piece of evidence for relevance to the claim segments
3. For each segment, determine its status based on the evidence
4. Consider source credibility (scores provided)
5. Generate a final verdict

VERDICT OPTIONS:
- TRUE: Evidence strongly supports the claim (multiple credible sources agree)
- FALSE: Evidence contradicts the claim (credible sources disagree with claim)
- PARTIALLY_TRUE: Some aspects are supported, others are not or lack evidence
- UNVERIFIABLE: Insufficient evidence to make a determination

CLAIM SEGMENT STATUS OPTIONS:
- VALID: Evidence confirms this part of the claim
- INVALID: Evidence contradicts this part of the claim
- PARTIALLY_VALID: Some evidence supports, but with caveats
- PARTIALLY_INVALID: Some evidence contradicts, but not completely
- UNKNOWN: Insufficient evidence to determine

Return ONLY valid JSON (no markdown, no extra text):
{{
    "verdict": "TRUE|FALSE|PARTIALLY_TRUE|UNVERIFIABLE",
    "confidence": 0.85,
    "rationale": "Brief explanation of why this verdict was reached",
    "claim_breakdown": [
        {{
            "claim_segment": "The specific part of the claim being evaluated",
            "status": "VALID|INVALID|PARTIALLY_VALID|PARTIALLY_INVALID|UNKNOWN",
            "supporting_fact": "The fact from evidence that supports or contradicts this segment",
            "source_url": "https://source.url.of.the.fact"
        }}
    ],
    "evidence_map": [
        {{
            "evidence_id": 0,
            "statement": "The evidence statement",
            "relevance": "SUPPORTS|CONTRADICTS|NEUTRAL",
            "relevance_score": 0.9,
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
        # CLAIM-SEGMENT RETRIEVAL: Query VDB for each claim segment
        # ================================================================
        segment_evidence = await self._retrieve_segment_evidence(claim, top_k=3)

        # Merge segment evidence with ranked evidence (dedup by statement)
        enriched_evidence = self._merge_evidence(ranked_evidence, segment_evidence)

        logger.info(
            f"[VerdictGenerator] Evidence: {len(ranked_evidence)} ranked + "
            f"{len(segment_evidence)} segment-specific = {len(enriched_evidence)} total"
        )

        # Take more evidence to include segment-specific facts
        # Use top_k + segment evidence count, capped at 10 to avoid prompt bloat
        evidence_limit = min(len(enriched_evidence), max(top_k, len(ranked_evidence) + len(segment_evidence)), 10)
        top_evidence = enriched_evidence[:evidence_limit]

        # Format evidence for prompt
        evidence_text = self._format_evidence_for_prompt(top_evidence)

        # Build RAG prompt
        prompt = VERDICT_GENERATION_PROMPT.format(claim=claim, evidence_text=evidence_text)

        try:
            # HIGH priority - verdict generation is critical
            result = await self.llm_service.ainvoke(prompt, response_format="json", priority=LLMPriority.HIGH)

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

        return {
            "verdict": verdict_str,
            "confidence": confidence,
            "rationale": rationale,
            "claim_breakdown": claim_breakdown,
            "evidence_map": evidence_map,
            "key_findings": key_findings,
            "claim": claim,
            "evidence_count": len(evidence),
        }

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

    async def _retrieve_segment_evidence(self, claim: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query VDB for each claim segment independently.

        This ensures that facts ingested in previous runs are found
        even when full-claim semantic similarity favors other topics.
        """
        segments = self._split_claim_into_segments(claim)
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
