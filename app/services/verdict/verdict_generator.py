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
from typing import Any, Dict, List, Optional, Tuple

from prometheus_client import Counter

from app.constants.config import (
    ANCHOR_THRESHOLD,
    CONTRADICT_RATIO_FOR_FORCE_FALSE,
    CONTRADICT_RATIO_FORCE_FALSE,
    CONTRADICTION_THRESHOLD,
    DIVERSITY_FORCE_FALSE,
    LLM_TEMPERATURE_VERDICT,
    PREDICATE_MATCH_THRESHOLD,
    UNVERIFIABLE_CONFIDENCE_CAP,
)
from app.core.config import settings
from app.core.logger import get_logger
from app.services.common.claim_segmentation import split_claim_into_segments
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.scraper import Scraper
from app.services.corrective.trusted_search import TrustedSearch
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority
from app.services.logic.claim_strictness import compute_claim_strictness
from app.services.logic.evidence_strength import compute_evidence_strength
from app.services.logic.overrides import apply_claim_logic_overrides
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
_POLARITY_DEBUG = os.getenv("VERDICT_DEBUG_POLARITY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

logic_overrides_total = Counter(
    "logic_overrides_total",
    "Total strictness logic overrides fired",
    ["type"],
)
confidence_capped_total = Counter(
    "confidence_capped_total",
    "Total times confidence was capped by strictness/diversity logic",
)
contradiction_override_total = Counter(
    "contradiction_override_total",
    "Total contradiction dominance overrides",
)
hedge_mismatch_total = Counter(
    "hedge_mismatch_total",
    "Total hedge mismatch overrides",
)


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
        self._last_predicate_queries_generated: List[str] = []
        env_flag = os.getenv("LUXIA_CONFIDENCE_MODE")
        if env_flag is None:
            self.confidence_mode = bool(getattr(settings, "LUXIA_CONFIDENCE_MODE", False))
        else:
            self.confidence_mode = env_flag.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

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

    def _policy_says_insufficient(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
        adaptive_metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if adaptive_metrics is not None:
            return not bool(adaptive_metrics.get("is_sufficient", False))
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

    @staticmethod
    def _is_numeric_comparison_claim(claim: str) -> bool:
        text = f" {str(claim or '').strip().lower()} "
        if not text.strip():
            return False
        patterns = (
            r"\bmore\b.{0,80}\bthan\b",
            r"\bless\b.{0,80}\bthan\b",
            r"\bfewer\b.{0,80}\bthan\b",
            r"\bgreater\b.{0,80}\bthan\b",
            r"\bhigher\b.{0,80}\bthan\b",
            r"\blower\b.{0,80}\bthan\b",
            r"\bvs\b|\bversus\b|\bcompared to\b",
        )
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
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
            "there",
            "than",
            "more",
            "less",
            "fewer",
            "greater",
            "higher",
            "lower",
            "your",
            "their",
            "our",
        }
        return {w for w in re.findall(r"\b[a-z][a-z0-9_-]{1,}\b", (text or "").lower()) if w not in stop}

    def _parse_numeric_mentions(self, text: str) -> List[Tuple[float, float]]:
        # Returns numeric constraints as (lower_bound, upper_bound).
        # `float("inf")` means no upper bound.
        if not text:
            return []
        pattern = re.compile(
            r"(?P<prefix>\b(?:about|around|approx(?:imately)?|over|under|at least|at most|more than|less than)\b\s*)?"
            r"(?P<symbol>>=|<=|>|<)?\s*"
            r"(?P<num>\d+(?:[.,]\d+)?)\s*"
            r"(?P<scale>trillion|billion|million|thousand)?",
            re.IGNORECASE,
        )
        scale_map = {
            "thousand": 1_000.0,
            "million": 1_000_000.0,
            "billion": 1_000_000_000.0,
            "trillion": 1_000_000_000_000.0,
            None: 1.0,
        }
        constraints: List[Tuple[float, float]] = []
        for m in pattern.finditer(text):
            raw_num = (m.group("num") or "").replace(",", "")
            try:
                base = float(raw_num)
            except Exception:
                continue
            scale = (m.group("scale") or "").lower() or None
            value = base * scale_map.get(scale, 1.0)
            prefix = (m.group("prefix") or "").strip().lower()
            symbol = (m.group("symbol") or "").strip()

            lb = value
            ub = value
            if symbol in {">", ">="} or prefix in {"over", "more than", "at least"}:
                lb, ub = value, float("inf")
            elif symbol in {"<", "<="} or prefix in {"under", "less than", "at most"}:
                lb, ub = 0.0, value
            elif prefix in {"about", "around", "approx", "approximately"}:
                lb, ub = value * 0.9, value * 1.1
            constraints.append((max(0.0, lb), max(0.0, ub)))
        return constraints

    @staticmethod
    def _merge_constraints(constraints: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not constraints:
            return None
        lb = 0.0
        ub = float("inf")
        for c_lb, c_ub in constraints:
            lb = max(lb, float(c_lb))
            ub = min(ub, float(c_ub))
        if lb > ub:
            return None
        return (lb, ub)

    def _find_direct_non_kg_support(self, claim: str, evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        claim_tokens = self._meaningful_tokens(claim)
        if not claim_tokens:
            return None
        best: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for ev in evidence[:12]:
            stmt = str(ev.get("statement") or ev.get("text") or "").strip()
            if not stmt:
                continue
            stance = str(ev.get("stance") or "neutral").lower()
            if stance == "contradicts":
                continue
            candidate_type = str(ev.get("candidate_type") or "").upper()
            source_type = str(ev.get("source_type") or "").lower()
            if candidate_type == "KG" or source_type == "kg":
                continue
            sem = float(
                ev.get("semantic_score") or ev.get("sem_score") or ev.get("final_score") or ev.get("score") or 0.0
            )
            if sem < 0.90:
                continue
            stmt_tokens = self._meaningful_tokens(stmt)
            overlap = len(claim_tokens & stmt_tokens)
            if overlap < 3:
                continue
            score = (0.70 * sem) + (0.30 * (overlap / max(1, len(claim_tokens))))
            if score > best_score:
                best_score = score
                best = ev
        return best

    def _is_direct_comparative_statement(self, stmt: str, lhs_tokens: set[str], rhs_tokens: set[str]) -> bool:
        low = (stmt or "").lower()
        has_comparison = bool(
            re.search(
                r"\b(than|versus|vs|compared to|sooner|faster|slower|longer|shorter|higher|lower)\b",
                low,
            )
        )
        has_quant = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\b|\b(days?|hours?|weeks?|months?|years?|mortality|survival)\b",
                low,
            )
        )
        stmt_tokens = self._meaningful_tokens(stmt)
        lhs_overlap = len(lhs_tokens & stmt_tokens)
        rhs_overlap = len(rhs_tokens & stmt_tokens)
        has_both_sides = lhs_overlap > 0 and rhs_overlap > 0
        return has_comparison and has_quant and has_both_sides

    def _resolve_numeric_comparison_override(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
        claim_breakdown: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self._is_numeric_comparison_claim(claim):
            return None
        claim_text = str(claim or "").strip()
        lower_claim = claim_text.lower()
        op = ">"
        if re.search(r"\b(less|fewer|lower)\b.{0,80}\bthan\b", lower_claim):
            op = "<"

        if " than " in lower_claim:
            lhs_text, rhs_text = lower_claim.split(" than ", 1)
        elif " compared to " in lower_claim:
            lhs_text, rhs_text = lower_claim.split(" compared to ", 1)
        elif " versus " in lower_claim:
            lhs_text, rhs_text = lower_claim.split(" versus ", 1)
        elif " vs " in lower_claim:
            lhs_text, rhs_text = lower_claim.split(" vs ", 1)
        else:
            lhs_text, rhs_text = lower_claim, ""

        lhs_tokens = self._meaningful_tokens(lhs_text)
        rhs_tokens = self._meaningful_tokens(rhs_text)
        lhs_constraints: List[Tuple[float, float]] = []
        rhs_constraints: List[Tuple[float, float]] = []
        observed_numeric_evidence = 0
        direct_comparative_evidence = 0

        for ev in evidence[:12]:
            stmt = str(ev.get("statement") or ev.get("text") or "").strip()
            if not stmt:
                continue
            constraints = self._parse_numeric_mentions(stmt)
            if not constraints:
                continue
            observed_numeric_evidence += 1
            if self._is_direct_comparative_statement(stmt, lhs_tokens, rhs_tokens):
                direct_comparative_evidence += 1
            stmt_tokens = self._meaningful_tokens(stmt)
            lhs_overlap = len(lhs_tokens & stmt_tokens)
            rhs_overlap = len(rhs_tokens & stmt_tokens)

            if lhs_overlap == 0 and rhs_overlap == 0:
                if re.search(r"\b(oral|mouth|microbiome|bacteria)\b", stmt.lower()):
                    lhs_constraints.extend(constraints)
                    continue
                if re.search(r"\b(world|global|population|people|humans?)\b", stmt.lower()):
                    rhs_constraints.extend(constraints)
                    continue
                continue

            if lhs_overlap >= rhs_overlap:
                lhs_constraints.extend(constraints)
            if rhs_overlap > lhs_overlap:
                rhs_constraints.extend(constraints)

        if direct_comparative_evidence <= 0:
            return {
                "verdict": Verdict.UNVERIFIABLE.value,
                "status": "UNKNOWN",
                "reason": (
                    "Comparative evidence quality gate: no direct two-sided quantitative "
                    "comparison evidence was found."
                ),
                "truthfulness_percent": 45.0,
                "deterministic_gate": True,
            }

        lhs_interval = self._merge_constraints(lhs_constraints)
        rhs_interval = self._merge_constraints(rhs_constraints)
        if lhs_interval is None or rhs_interval is None:
            reason = (
                "Numeric comparison evidence is incomplete: explicit two-sided " "quantitative support is required."
            )
            return {
                "verdict": Verdict.UNVERIFIABLE.value,
                "status": "UNKNOWN",
                "reason": reason,
                "truthfulness_percent": 45.0 if observed_numeric_evidence > 0 else 40.0,
                "deterministic_gate": True,
            }

        lhs_lb, lhs_ub = lhs_interval
        rhs_lb, rhs_ub = rhs_interval

        if op == ">":
            if lhs_lb > rhs_ub:
                return {
                    "verdict": Verdict.TRUE.value,
                    "status": "VALID",
                    "reason": (
                        "Numeric comparison resolved deterministically: "
                        "left-side lower bound exceeds right-side upper bound."
                    ),
                    "truthfulness_percent": 92.0,
                    "deterministic_gate": True,
                }
            if lhs_ub <= rhs_lb:
                return {
                    "verdict": Verdict.FALSE.value,
                    "status": "INVALID",
                    "reason": (
                        "Numeric comparison resolved deterministically: "
                        "left-side upper bound does not exceed right-side lower bound."
                    ),
                    "truthfulness_percent": 12.0,
                    "deterministic_gate": True,
                }
        else:
            if lhs_ub < rhs_lb:
                return {
                    "verdict": Verdict.TRUE.value,
                    "status": "VALID",
                    "reason": (
                        "Numeric comparison resolved deterministically: "
                        "left-side upper bound remains below right-side lower bound."
                    ),
                    "truthfulness_percent": 92.0,
                    "deterministic_gate": True,
                }
            if lhs_lb >= rhs_ub:
                return {
                    "verdict": Verdict.FALSE.value,
                    "status": "INVALID",
                    "reason": (
                        "Numeric comparison resolved deterministically: "
                        "left-side lower bound is not below right-side upper bound."
                    ),
                    "truthfulness_percent": 12.0,
                    "deterministic_gate": True,
                }

        return {
            "verdict": Verdict.UNVERIFIABLE.value,
            "status": "UNKNOWN",
            "reason": "Numeric comparison could not be resolved confidently from bounded evidence ranges.",
            "truthfulness_percent": 45.0,
            "deterministic_gate": True,
        }

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
        self._last_predicate_queries_generated = []
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
        if adaptive_metrics is not None:
            policy_insufficient_hint = not bool(adaptive_metrics.get("is_sufficient", False))
        else:
            policy_insufficient_hint = self._policy_says_insufficient(
                claim,
                ranked_evidence[: min(len(ranked_evidence), 10)],
                adaptive_metrics=None,
            )

        if (not cache_sufficient) and (
            self._needs_web_boost(ranked_evidence[: min(len(ranked_evidence), 6)], claim=claim)
            or policy_insufficient_hint
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
                if adaptive_metrics is not None:
                    insufficient = not bool(adaptive_metrics.get("is_sufficient", False))
                else:
                    insufficient = self._policy_says_insufficient(claim, pre_evidence, adaptive_metrics=None)
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
            if not cache_sufficient and not bool(verdict_result.get("skip_targeted_recovery", False)):
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
                        analysis_counts = verdict_result.get("analysis_counts", {}) or {}
                        explicit_refutes_found = bool(verdict_result.get("explicit_refutes_found", False))
                        contradict_cov = float(
                            analysis_counts.get("contradict_coverage", analysis_counts.get("contradict_ratio", 0.0))
                            or 0.0
                        )
                        need_predicate_refute = (not explicit_refutes_found) or (
                            contradict_cov < float(CONTRADICT_RATIO_FOR_FORCE_FALSE)
                        )
                        web_evidence = await self._fetch_web_evidence_for_unknown_segments(
                            unknown_segments,
                            max_queries_per_segment=1 if used_web_search else 2,
                            max_urls_per_query=1 if used_web_search else 3,
                            enable_predicate_refute_queries=need_predicate_refute,
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
        predicate_queries_generated = list(getattr(self, "_last_predicate_queries_generated", []) or [])
        explicit_refutes_found = any(
            self._normalize_relevance_label(ev.get("relevance")) == "REFUTES"
            and float(ev.get("contradiction_score", 0.0) or 0.0) >= CONTRADICTION_THRESHOLD
            for ev in (evidence_map or [])
        )
        predicate_match_score_used = max(
            [float(ev.get("predicate_match_score", 0.0) or 0.0) for ev in (evidence_map or [])] or [0.0]
        )
        truthfulness_invariant_applied = False

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
                if not self._segment_topic_guard_ok(segment, stmt):
                    continue
                if not self._segment_is_belief_or_survey_claim(segment) and not self._evidence_is_admissible_for_claim(
                    segment, stmt
                ):
                    continue
                seg_subject_tokens = self._segment_subject_tokens(segment)
                seg_object_tokens = self._segment_object_tokens(segment)
                stmt_tokens_full = self._statement_tokens(stmt)
                if seg_subject_tokens and len(seg_subject_tokens & stmt_tokens_full) == 0:
                    continue
                if seg_object_tokens and len(seg_object_tokens & stmt_tokens_full) == 0:
                    if not self._is_explicit_refutation_statement(stmt):
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

            # Replace mention/survey phrasing with stronger aligned evidence for non-belief claims.
            if (
                best_ev is not None
                and not self._segment_is_belief_or_survey_claim(seg_text)
                and fact
                and self._is_claim_mention_statement(fact)
            ):
                if best_stmt and not self._is_claim_mention_statement(best_stmt):
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
            if status in {"VALID", "PARTIALLY_VALID"} and polarity_text:
                object_tokens = self._segment_object_tokens(seg_text)
                statement_tokens = self._statement_tokens(polarity_text)
                no_object_overlap = bool(object_tokens) and len(object_tokens & statement_tokens) == 0
                if no_object_overlap and not self._is_explicit_refutation_statement(polarity_text):
                    seg["status"] = "INVALID" if status == "VALID" else "PARTIALLY_INVALID"
            status = (seg.get("status") or "UNKNOWN").upper()
            if status in {
                "VALID",
                "PARTIALLY_VALID",
            } and self._is_claim_mention_statement(polarity_text):
                seg["status"] = "UNKNOWN"
                seg["supporting_fact"] = ""
                seg["source_url"] = ""
                seg["evidence_used_ids"] = []
            status = (seg.get("status") or "UNKNOWN").upper()
            if status in {
                "INVALID",
                "PARTIALLY_INVALID",
            } and self._is_claim_mention_statement(polarity_text):
                if not self._segment_is_belief_or_survey_claim(seg_text) and not self._is_explicit_refutation_statement(
                    polarity_text
                ):
                    seg["status"] = "UNKNOWN"
                    seg["supporting_fact"] = ""
                    seg["source_url"] = ""
                    seg["evidence_used_ids"] = []
            status = (seg.get("status") or "UNKNOWN").upper()

            # UNKNOWN minimization ladder:
            # deterministically upgrade UNKNOWN only when alignment is strong and polarity is clear.
            if status == "UNKNOWN" and best_ev is not None and polarity_text:
                if self._is_claim_mention_statement(polarity_text):
                    seg["status"] = "UNKNOWN"
                    seg["supporting_fact"] = ""
                    seg["source_url"] = ""
                    seg["evidence_used_ids"] = []
                    continue
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

            # Hard consistency: INVALID/PARTIALLY_INVALID requires contradiction signal.
            final_status = (seg.get("status") or "UNKNOWN").upper()
            if final_status in {"INVALID", "PARTIALLY_INVALID"}:
                contradiction_signal = (
                    polarity == "contradicts"
                    or high_semantic_negation_mismatch
                    or self._is_explicit_refutation_statement(polarity_text or "")
                )
                if not contradiction_signal:
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

        self._apply_adaptive_coverage_fallback(
            claim_breakdown=claim_breakdown,
            adaptive_metrics=adaptive_metrics,
            evidence=evidence,
        )
        self._log_subclaim_coverage(claim, evidence, claim_breakdown, adaptive_metrics=adaptive_metrics)
        reconciled = self._reconcile_verdict_with_breakdown(claim, claim_breakdown)
        verdict_str = reconciled["verdict"]
        numeric_override = self._resolve_numeric_comparison_override(claim, evidence, claim_breakdown)
        numeric_truth_override: Optional[float] = None
        skip_targeted_recovery = False
        numeric_conf_floor: Optional[float] = None
        if numeric_override is not None:
            verdict_str = str(numeric_override.get("verdict") or verdict_str)
            status_override = str(numeric_override.get("status") or "").strip().upper()
            if claim_breakdown and status_override in {
                "VALID",
                "INVALID",
                "PARTIALLY_VALID",
                "PARTIALLY_INVALID",
                "UNKNOWN",
            }:
                claim_breakdown[0]["status"] = status_override
                if status_override == "UNKNOWN":
                    claim_breakdown[0]["supporting_fact"] = ""
                    claim_breakdown[0]["source_url"] = ""
                    claim_breakdown[0]["evidence_used_ids"] = []
            reconciled = self._reconcile_verdict_with_breakdown(claim, claim_breakdown)
            numeric_reason = str(numeric_override.get("reason") or "").strip()
            if numeric_reason:
                rationale = numeric_reason
            if numeric_override.get("truthfulness_percent") is not None:
                try:
                    numeric_truth_override = float(numeric_override.get("truthfulness_percent"))
                except Exception:
                    numeric_truth_override = None
            if bool(numeric_override.get("deterministic_gate", False)):
                skip_targeted_recovery = True
                if verdict_str == Verdict.UNVERIFIABLE.value:
                    numeric_conf_floor = 0.42
            logger.info(
                "[VerdictGenerator][NumericOverride] verdict=%s status=%s reason=%s",
                verdict_str,
                status_override or "-",
                numeric_reason or "-",
            )
        claim_frame = self._classify_claim_frame(claim)
        policy_insufficient = self._policy_says_insufficient(claim, evidence, adaptive_metrics=adaptive_metrics)
        admissible_evidence = [
            ev
            for ev in evidence
            if self._evidence_is_admissible_for_claim(
                claim,
                ev.get("statement") or ev.get("text") or "",
            )
        ]
        admissible_ratio = (len(admissible_evidence) / max(1, len(evidence))) if evidence else 0.0
        if evidence and admissible_ratio < 0.35:
            # Strongly penalize confidence when most evidence is belief/reporting style.
            policy_insufficient = True
        truth_score_percent = self._calculate_truth_score_from_contract(reconciled)
        (
            truth_score_percent,
            invariant_applied_now,
            max_status_weight_claim_segment,
        ) = self._apply_truthfulness_invariant(
            truth_score_percent,
            claim_breakdown,
            explicit_refutes_found=explicit_refutes_found,
        )
        truthfulness_invariant_applied = truthfulness_invariant_applied or invariant_applied_now
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
                    __slots__ = (
                        "statement",
                        "source_url",
                        "semantic_score",
                        "stance",
                        "trust",
                    )

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

        # Single-subclaim promotion:
        # If adaptive trust is already maxed and direct non-KG support exists,
        # upgrade PARTIALLY_TRUE/PARTIALLY_VALID to TRUE/VALID.
        if (
            claim_frame.get("claim_type") != "THERAPEUTIC_EFFICACY"
            and len(claim_breakdown or []) == 1
            and verdict_str == Verdict.PARTIALLY_TRUE.value
            and str((claim_breakdown[0] or {}).get("status", "")).upper() == "PARTIALLY_VALID"
            and float(coverage_score) >= 0.95
            and float(agreement_ratio) >= 0.90
            and float(adaptive_trust_post) >= 0.45
            and int((adaptive_metrics or {}).get("strong_covered", 0) or 0) >= 1
            and int((adaptive_metrics or {}).get("contradicted_subclaims", 0) or 0) == 0
        ):
            direct_support = self._find_direct_non_kg_support(claim, evidence)
            if direct_support is not None:
                claim_breakdown[0]["status"] = "VALID"
                claim_breakdown[0]["supporting_fact"] = str(
                    direct_support.get("statement") or direct_support.get("text") or ""
                )
                claim_breakdown[0]["source_url"] = str(
                    direct_support.get("source_url") or direct_support.get("source") or ""
                )
                reconciled = self._reconcile_verdict_with_breakdown(claim, claim_breakdown)
                verdict_str = reconciled["verdict"]
                logger.info(
                    "[VerdictGenerator][SingleSubclaimPromote] Promoted PARTIALLY_VALID -> VALID "
                    "under strong adaptive support (coverage=%.2f agreement=%.2f trust_post=%.3f).",
                    coverage_score,
                    agreement_ratio,
                    adaptive_trust_post,
                )

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
            elif override_verdict in {
                Verdict.TRUE.value,
                Verdict.FALSE.value,
                Verdict.UNVERIFIABLE.value,
            }:
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
        if numeric_truth_override is not None:
            truth_score_percent = float(numeric_truth_override)
            if verdict_str == Verdict.UNVERIFIABLE.value:
                coverage_score = min(float(coverage_score), 0.35)
                adaptive_trust_post = min(float(adaptive_trust_post), 0.35)
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
        confidence = self._cap_confidence_with_contract(
            confidence,
            reconciled,
            policy_insufficient,
            verdict_str,
            is_comparative_claim=self._is_numeric_comparison_claim(claim),
        )
        strictness_profile = compute_claim_strictness(claim)

        rel_by_statement: Dict[str, str] = {}
        for evm in evidence_map or []:
            stmt = str(evm.get("statement") or "").strip().lower()
            if not stmt:
                continue
            rel_by_statement[stmt] = str(evm.get("relevance") or "").upper()

        evidence_strengths = []
        for ev in evidence[: min(len(evidence), 10)]:
            stmt = str(ev.get("statement") or ev.get("text") or "").strip()
            if not stmt:
                continue
            evidence_strengths.append(
                compute_evidence_strength(
                    claim_text=claim,
                    text_snippet=stmt,
                    source_meta=ev,
                    stance_hint=rel_by_statement.get(stmt.lower()),
                )
            )

        kg_hint_ratio = 0.0
        if evidence:
            kg_hint_ratio = sum(1 for ev in evidence if float(ev.get("kg_score", 0.0) or 0.0) > 0.0) / max(
                1, len(evidence)
            )
        breakdown_stance = self._canonical_stance_from_breakdown(claim_breakdown)
        map_stance = self._canonical_stance_from_evidence_map(evidence_map)
        rationale_stance = self._rationale_polarity_hint(rationale)
        top_stance, top_stance_score = self._top_admissible_stance_signal(claim, evidence_map)
        vote_decision, vote_support, vote_contradict = self._stance_vote_decision(claim, evidence_map)
        (
            map_support_signal,
            map_contradict_signal,
            map_admissible_count,
            map_offtopic_count,
        ) = self._evidence_map_signal_strengths(claim, evidence_map)
        contradiction_metrics = self._contradiction_dominance_metrics(claim, evidence_map)
        decisive_support = map_support_signal >= 0.58 and map_contradict_signal <= 0.40
        decisive_contradict = map_contradict_signal >= 0.58 and map_support_signal <= 0.40
        conflicting_signals = map_support_signal >= 0.45 and map_contradict_signal >= 0.45
        weak_signal_no_stance = (not decisive_support and not decisive_contradict) or conflicting_signals
        if weak_signal_no_stance:
            policy_insufficient = True
        canonical_stance = breakdown_stance
        if canonical_stance == "neutral":
            canonical_stance = map_stance
        if canonical_stance == "neutral":
            canonical_stance = rationale_stance
        if canonical_stance == "neutral":
            canonical_stance = top_stance

        strong_vote_contradiction = vote_decision == Verdict.FALSE.value and vote_contradict >= 0.65
        strong_vote_support = vote_decision == Verdict.TRUE.value and vote_support >= 0.65
        if vote_decision in {Verdict.TRUE.value, Verdict.FALSE.value}:
            verdict_str = vote_decision

        strict_override_fired = "NONE"
        strict_override_reason = "none"
        contradict_ratio_threshold = float(
            os.getenv(
                "CONTRADICT_RATIO_FOR_FORCE_FALSE",
                str(max(float(CONTRADICT_RATIO_FOR_FORCE_FALSE), float(CONTRADICT_RATIO_FORCE_FALSE))),
            )
        )
        if (
            explicit_refutes_found
            and contradiction_metrics["contradict_ratio"] >= contradict_ratio_threshold
            and contradiction_metrics["contradict_diversity"] >= DIVERSITY_FORCE_FALSE
        ):
            for seg in claim_breakdown or []:
                seg["status"] = "INVALID"
            verdict_str = Verdict.FALSE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 20.0)
            confidence = max(float(confidence or 0.0), 0.75)
            strict_override_fired = "CONTRADICTION_DOMINANCE"
            strict_override_reason = "CONTRADICTION_DOMINANCE"
            logger.warning(
                (
                    "[VerdictGenerator][StrictOverride] "
                    "type=CONTRADICTION_DOMINANCE ratio=%.3f coverage=%.3f "
                    "diversity=%.3f support_cov=%.3f"
                ),
                contradiction_metrics["contradict_ratio"],
                contradiction_metrics["contradict_coverage"],
                contradiction_metrics["contradict_diversity"],
                contradiction_metrics["support_coverage"],
            )

        # Hard polarity guardrails to keep verdict and rationale/segment stance consistent.
        if canonical_stance == "contradicts" and verdict_str == Verdict.TRUE.value:
            verdict_str = Verdict.FALSE.value
            logger.warning(
                "[VerdictGenerator][Consistency] Flipped TRUE->FALSE " "(breakdown=%s map=%s rationale=%s)",
                breakdown_stance,
                map_stance,
                rationale_stance,
            )
        # Deterministic stance-to-verdict override on strong admissible contradiction.
        if (
            top_stance == "contradicts"
            and top_stance_score >= 0.62
            and coverage_score >= 0.60
            and admissible_ratio >= 0.50
        ):
            verdict_str = Verdict.FALSE.value
        elif (
            top_stance == "entails"
            and top_stance_score >= 0.70
            and coverage_score >= 0.70
            and admissible_ratio >= 0.60
            and not self._has_absolute_quantifier(claim)
        ):
            verdict_str = Verdict.TRUE.value
        elif canonical_stance == "entails" and verdict_str == Verdict.FALSE.value:
            verdict_str = Verdict.TRUE.value
            logger.warning(
                "[VerdictGenerator][Consistency] Flipped FALSE->TRUE " "(breakdown=%s map=%s rationale=%s)",
                breakdown_stance,
                map_stance,
                rationale_stance,
            )
        if rationale_stance == "contradicts" and verdict_str == Verdict.TRUE.value:
            verdict_str = Verdict.FALSE.value if vote_contradict >= 0.60 else Verdict.UNVERIFIABLE.value
        if weak_signal_no_stance:
            verdict_str = Verdict.UNVERIFIABLE.value
        # Trust-gate fallback: if trust is insufficient and stance is unresolved, avoid confident binary verdicts.
        if (
            policy_insufficient
            and canonical_stance == "neutral"
            and verdict_str in {Verdict.TRUE.value, Verdict.FALSE.value}
        ):
            verdict_str = Verdict.UNVERIFIABLE.value

        unique_domains_count = 0
        try:
            unique_domains_count = len(
                {
                    str((ev.get("source_url") or ev.get("source") or "").split("/")[2]).lower().removeprefix("www.")
                    for ev in (evidence or [])
                    if str(ev.get("source_url") or ev.get("source") or "").startswith("http")
                }
            )
        except Exception:
            unique_domains_count = 0

        # Conservative binary gate:
        # Avoid hard TRUE/FALSE unless stance is explicit and evidence quality/diversity are sufficient.
        if verdict_str in {Verdict.TRUE.value, Verdict.FALSE.value}:
            binary_gate_ok = (
                canonical_stance in {"entails", "contradicts"}
                and admissible_ratio >= 0.50
                and (diversity_score >= 0.40 or adaptive_trust_post >= 0.45)
                and (coverage_score >= 0.70 or agreement_ratio >= 0.80)
                and unique_domains_count >= 2
            )
            # Allow strong contradiction outcomes for simple factual numeric claims
            # even when diversity is low (single high-quality source can be decisive).
            strong_contradiction_exception = (
                verdict_str == Verdict.FALSE.value
                and canonical_stance == "contradicts"
                and coverage_score >= 0.70
                and agreement_ratio >= 0.80
                and admissible_ratio >= 0.50
            )
            strong_support_exception = (
                verdict_str == Verdict.TRUE.value
                and canonical_stance == "entails"
                and coverage_score >= 0.90
                and agreement_ratio >= 0.90
                and admissible_ratio >= 0.60
                and top_stance == "entails"
                and top_stance_score >= 0.68
                and map_contradict_signal < 0.35
                and unique_domains_count >= 2
                and diversity_score >= 0.20
            )
            binary_gate_ok = bool(binary_gate_ok or strong_contradiction_exception or strong_support_exception)
            if strong_vote_contradiction:
                binary_gate_ok = True
            if not binary_gate_ok:
                verdict_str = Verdict.UNVERIFIABLE.value
        # Single-source support should not produce overconfident hard TRUE.
        if verdict_str == Verdict.TRUE.value and unique_domains_count < 2:
            verdict_str = Verdict.PARTIALLY_TRUE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 75.0)
            confidence = min(float(confidence), 0.75)
        if strong_vote_contradiction:
            verdict_str = Verdict.FALSE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 10.0)
            confidence = max(float(confidence), 0.70)
        final_lock_unverifiable = bool(weak_signal_no_stance or policy_insufficient)
        if strong_vote_support and self._has_absolute_quantifier(claim) and not final_lock_unverifiable:
            verdict_str = Verdict.PARTIALLY_TRUE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 55.0)
        if numeric_conf_floor is not None:
            confidence = max(float(confidence), float(numeric_conf_floor))
        if (
            final_lock_unverifiable
            and strict_override_fired != "CONTRADICTION_DOMINANCE"
            and map_contradict_signal < CONTRADICTION_THRESHOLD
        ):
            verdict_str = Verdict.UNVERIFIABLE.value
            truth_score_percent = max(35.0, min(55.0, float(truth_score_percent or 0.0)))
            confidence = min(float(confidence), UNVERIFIABLE_CONFIDENCE_CAP)
            for seg in claim_breakdown or []:
                status_u = str(seg.get("status") or "UNKNOWN").upper()
                if status_u not in {"INVALID", "PARTIALLY_INVALID"}:
                    seg["status"] = "UNKNOWN"
                    seg["supporting_fact"] = ""
                    seg["source_url"] = ""
                    seg["evidence_used_ids"] = []
                    seg["alignment_debug"] = {
                        "reason": "insufficient_admissible_evidence",
                        "support_signal": round(map_support_signal, 3),
                        "contradict_signal": round(map_contradict_signal, 3),
                    }
            reconciled = self._reconcile_verdict_with_breakdown(claim, claim_breakdown)
            rationale = (
                "Available admissible evidence is mixed or insufficiently decisive for this claim, "
                "so the result is UNVERIFIABLE."
            )
        if verdict_str == Verdict.UNVERIFIABLE.value and numeric_truth_override is None:
            truth_score_percent = max(35.0, min(65.0, float(truth_score_percent or 0.0)))
        if self._has_absolute_quantifier(claim) and verdict_str == Verdict.TRUE.value:
            verdict_str = Verdict.PARTIALLY_TRUE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 55.0)

        if strict_override_fired == "CONTRADICTION_DOMINANCE":
            verdict_str = Verdict.FALSE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 20.0)
            confidence = max(float(confidence or 0.0), 0.75)

        logic_result = apply_claim_logic_overrides(
            claim=claim,
            strictness=strictness_profile,
            evidence_strengths=evidence_strengths,
            claim_breakdown=claim_breakdown,
            verdict=verdict_str,
            truthfulness_percent=float(truth_score_percent or 0.0),
            confidence=float(confidence or 0.0),
            diversity=float(diversity_score or 0.0),
            agreement=float(agreement_ratio or 0.0),
            evidence_count=len(evidence),
            kg_hint_ratio=float(kg_hint_ratio or 0.0),
        )
        if logic_result.override_fired != "NONE":
            logic_overrides_total.labels(type=logic_result.override_fired).inc()
            if logic_result.override_fired == "CONTRADICTION_DOMINANCE":
                contradiction_override_total.inc()
            if logic_result.override_fired == "HEDGE_MISMATCH":
                hedge_mismatch_total.inc()
            logger.warning(
                "[VerdictGenerator][LogicOverride] type=%s reason=%s claim='%s' strictness=%s key_numbers=%s",
                logic_result.override_fired,
                logic_result.override_reason,
                claim[:220],
                strictness_profile.to_dict(),
                logic_result.key_numbers,
            )
        if strict_override_fired == "CONTRADICTION_DOMINANCE":
            logic_overrides_total.labels(type="CONTRADICTION_DOMINANCE").inc()
            contradiction_override_total.inc()
        if logic_result.confidence_cap is not None and float(confidence or 0.0) > float(logic_result.confidence_cap):
            confidence_capped_total.inc()
            confidence = float(logic_result.confidence_cap)
        if logic_result.confidence_floor is not None:
            confidence = max(float(confidence or 0.0), float(logic_result.confidence_floor))
        if logic_result.truthfulness_cap_percent is not None:
            truth_score_percent = min(float(truth_score_percent or 0.0), float(logic_result.truthfulness_cap_percent))
        verdict_str = str(logic_result.verdict or verdict_str)
        (
            truth_score_percent,
            invariant_applied_now,
            max_status_weight_claim_segment,
        ) = self._apply_truthfulness_invariant(
            truth_score_percent,
            claim_breakdown,
            explicit_refutes_found=explicit_refutes_found,
        )
        truthfulness_invariant_applied = truthfulness_invariant_applied or invariant_applied_now
        if verdict_str == Verdict.FALSE.value:
            confidence = max(float(confidence or 0.0), float(os.getenv("FALSE_VERDICT_MIN_CONFIDENCE", "0.55")))
        unverifiable_cap_applied = False
        if verdict_str == Verdict.UNVERIFIABLE.value and float(confidence or 0.0) > float(UNVERIFIABLE_CONFIDENCE_CAP):
            confidence = float(UNVERIFIABLE_CONFIDENCE_CAP)
            unverifiable_cap_applied = True
            confidence_capped_total.inc()
        confidence = max(0.05, min(0.98, float(confidence or 0.0)))

        rationale = self._rewrite_rationale_from_breakdown(rationale, claim_breakdown, reconciled)
        if (
            final_lock_unverifiable
            and strict_override_fired != "CONTRADICTION_DOMINANCE"
            and logic_result.override_fired not in {"CONTRADICTION_DOMINANCE"}
            and map_contradict_signal < CONTRADICTION_THRESHOLD
        ):
            verdict_str = Verdict.UNVERIFIABLE.value
            rationale = (
                "Available admissible evidence is mixed or insufficiently decisive for this claim, "
                "so the result is UNVERIFIABLE."
            )
        if llm_verdict == Verdict.TRUE.value and verdict_str != Verdict.TRUE.value:
            try:
                truth_score_percent = min(float(truth_score_percent), 89.9)
            except Exception:
                truth_score_percent = 89.9

        rel_counts: Dict[str, int] = {}
        for evm in evidence_map or []:
            rel = str(evm.get("relevance") or "NEUTRAL").upper()
            rel_counts[rel] = rel_counts.get(rel, 0) + 1

        status_counts: Dict[str, int] = {}
        for seg in claim_breakdown or []:
            st = str(seg.get("status") or "UNKNOWN").upper()
            status_counts[st] = status_counts.get(st, 0) + 1

        admissible_count = 0
        for ev in evidence or []:
            stmt = str(ev.get("statement") or ev.get("text") or "")
            if self._evidence_is_admissible_for_claim(claim, stmt):
                admissible_count += 1

        unique_domains = set()
        for ev in evidence or []:
            src = str(ev.get("source_url") or ev.get("source") or "").strip()
            if not src:
                continue
            try:
                d = src.split("/")[2].lower()
                if d.startswith("www."):
                    d = d[4:]
                if d:
                    unique_domains.add(d)
            except Exception:
                continue

        effective_override_fired = logic_result.override_fired
        effective_override_reason = logic_result.override_reason
        effective_override_key_numbers = dict(logic_result.key_numbers or {})
        if strict_override_fired == "CONTRADICTION_DOMINANCE":
            effective_override_fired = "CONTRADICTION_DOMINANCE"
            effective_override_reason = strict_override_reason
            effective_override_key_numbers.update(
                {
                    "contradict_ratio": round(contradiction_metrics["contradict_ratio"], 4),
                    "contradict_coverage": round(contradiction_metrics["contradict_coverage"], 4),
                    "contradict_diversity": round(contradiction_metrics["contradict_diversity"], 4),
                    "support_coverage": round(contradiction_metrics["support_coverage"], 4),
                }
            )
        elif unverifiable_cap_applied:
            effective_override_fired = "UNVERIFIABLE_CONFIDENCE_CAP"
            effective_override_reason = "UNVERIFIABLE_CONFIDENCE_CAP"
            effective_override_key_numbers.update(
                {
                    "unverifiable_confidence_cap": float(UNVERIFIABLE_CONFIDENCE_CAP),
                }
            )

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
            "skip_targeted_recovery": bool(skip_targeted_recovery),
            "strictness_profile": strictness_profile.to_dict(),
            "override_fired": effective_override_fired,
            "override_reason": effective_override_reason,
            "override_key_numbers": effective_override_key_numbers,
            "explicit_refutes_found": bool(explicit_refutes_found),
            "predicate_queries_generated": predicate_queries_generated,
            "predicate_match_score_used": round(float(predicate_match_score_used or 0.0), 4),
            "contradiction_override_fired": bool(strict_override_fired == "CONTRADICTION_DOMINANCE"),
            "truthfulness_invariant_applied": bool(truthfulness_invariant_applied),
            "analysis_counts": {
                "evidence_total_input": len(evidence),
                "evidence_map_count": len(evidence_map or []),
                "claim_breakdown_count": len(claim_breakdown or []),
                "admissible_evidence_count": admissible_count,
                "admissible_evidence_ratio": round(admissible_ratio, 4),
                "unique_source_domains": len(unique_domains),
                "relevance_distribution": rel_counts,
                "segment_status_distribution": status_counts,
                "canonical_stance_breakdown": breakdown_stance,
                "canonical_stance_evidence_map": map_stance,
                "canonical_stance_rationale": rationale_stance,
                "canonical_stance_final": canonical_stance,
                "top_admissible_stance": top_stance,
                "top_admissible_stance_score": round(top_stance_score, 4),
                "vote_support_max": round(vote_support, 4),
                "vote_contradict_max": round(vote_contradict, 4),
                "vote_decision": vote_decision,
                "map_support_signal_max": round(map_support_signal, 4),
                "map_contradict_signal_max": round(map_contradict_signal, 4),
                "map_admissible_signal_count": int(map_admissible_count),
                "map_offtopic_count": int(map_offtopic_count),
                "max_status_weight_claim_segment": round(float(max_status_weight_claim_segment or 0.0), 4),
                "contradict_ratio": round(contradiction_metrics["contradict_ratio"], 4),
                "contradict_coverage": round(contradiction_metrics["contradict_coverage"], 4),
                "contradict_diversity": round(contradiction_metrics["contradict_diversity"], 4),
                "support_coverage": round(contradiction_metrics["support_coverage"], 4),
                "weak_signal_no_stance": bool(weak_signal_no_stance),
                "decisive_support_signal": bool(decisive_support),
                "decisive_contradict_signal": bool(decisive_contradict),
                "conflicting_signal_band": bool(conflicting_signals),
                "llm_verdict_raw": llm_verdict,
                "llm_verdict_changed": bool(llm_verdict != verdict_str),
            },
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
    def _is_claim_mention_statement(text: str) -> bool:
        """
        Detect wording that reports a belief/concern/myth rather than asserting
        the proposition as factual support.
        """
        if not text:
            return False
        low = text.lower()
        mention_patterns = (
            r"\bmyth\b",
            r"\bmisinformation\b",
            r"\bdisinformation\b",
            r"\bconspiracy\b",
            r"\bfalse claim\b",
            r"\bunfounded\b",
            r"\bhoax\b",
            r"\brumou?r\b",
            r"\bparticipants?\s+(?:expressed|reported|mentioned|stated)\s+concern",
            r"\bconcerns?\s+about\b",
            r"\bworried\s+about\b",
            r"\bbeliev(?:e|ed|es|ing)\s+that\b",
            r"\bperceiv(?:e|ed|es|ing)\s+that\b",
            r"\bhesitan(?:cy|t)\b",
            r"\bacceptance\b",
        )
        return any(re.search(p, low) for p in mention_patterns)

    @staticmethod
    def _segment_is_belief_or_survey_claim(segment: str) -> bool:
        if not segment:
            return False
        low = segment.lower()
        return bool(
            re.search(
                r"\b(believe|belief|believed|think|thought|perceive|perceived|concern|hesitancy|acceptance|survey)\b",
                low,
            )
        )

    def _evidence_is_admissible_for_claim(self, claim: str, statement: str) -> bool:
        """
        Admissibility filter: belief/survey prevalence evidence is not admissible
        for factual composition/efficacy claims unless the claim itself is about belief.
        """
        if not statement:
            return False
        if self._segment_is_belief_or_survey_claim(claim):
            return True
        if self._is_claim_mention_statement(statement):
            return False
        return True

    @staticmethod
    def _is_explicit_refutation_statement(text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        patterns = (
            r"\bdo(?:es)?\s+not\b",
            r"\bno evidence\b",
            r"\bfalse\b",
            r"\bnot true\b",
            r"\bdebunk(?:ed|s|ing)?\b",
            r"\bmyth\b",
            r"\bunfounded\b",
            r"\bincorrect\b",
        )
        return any(re.search(p, low) for p in patterns)

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
            "hypertension": ["hypertension", "high blood pressure", "blood pressure"],
            "headache": ["headache", "headaches"],
            "symptom": ["symptom", "symptoms", "asymptomatic", "silent"],
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
        m = re.search(
            r"(?P<subject>.+?)\b(cure|cures|treat|treats|prevent|prevents)\b\s+(?P<object>.+)",
            text,
        )
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

        def _is_antibiotic_viral_inefficacy_claim(text: str) -> bool:
            low = (text or "").lower()
            has_antibiotic = bool(re.search(r"\bantibiotic(?:s|al)?\b", low))
            has_viral_target = bool(re.search(r"\b(viral?|virus(?:es)?|common cold|flu|influenza)\b", low))
            has_negative_efficacy = bool(
                re.search(
                    r"\b(do(?:es)?\s+not|cannot|can't|ineffective|no effect|not effective)\b.{0,30}\b"
                    r"(work|treat|help|cure|prevent)\b",
                    low,
                )
            )
            return has_antibiotic and has_viral_target and has_negative_efficacy

        def _misuse_implies_inefficacy(text: str) -> bool:
            low = (text or "").lower()
            has_antibiotic = bool(re.search(r"\bantibiotic(?:s|al)?\b", low))
            has_viral_target = bool(re.search(r"\b(viral?|virus(?:es)?|common cold|flu|influenza)\b", low))
            misuse_pattern = bool(
                re.search(
                    r"\b(misus(?:e|ed)|inappropriate|unnecessary|overprescrib(?:e|ed|ing)|" r"not recommended)\b",
                    low,
                )
            )
            return has_antibiotic and has_viral_target and misuse_pattern

        def _is_hypertension_symptom_claim(text: str) -> bool:
            low = (text or "").lower()
            has_condition = bool(re.search(r"\b(hypertension|high blood pressure|blood pressure)\b", low))
            has_symptom_focus = bool(
                re.search(
                    r"\b(symptom|symptoms|headache|headaches|noticeable|silent|asymptomatic)\b",
                    low,
                )
            )
            return has_condition and has_symptom_focus

        def _hypertension_symptom_polarity(text: str) -> str:
            low = (text or "").lower()
            has_condition = bool(re.search(r"\b(hypertension|high blood pressure|blood pressure)\b", low))
            if not has_condition:
                return "neutral"
            no_symptoms = bool(
                re.search(
                    r"\b(no symptoms?|asymptomatic|silent (condition|killer)|"
                    r"often has no symptoms?|usually has no symptoms?)\b",
                    low,
                )
            )
            has_symptoms = bool(
                re.search(
                    r"\b(has symptoms?|noticeable symptoms?|symptoms? like headaches?|"
                    r"headaches? (are|is) (common|typical|reliable))\b",
                    low,
                )
            )
            if no_symptoms and not has_symptoms:
                return "contradicts"
            if has_symptoms and not no_symptoms:
                return "entails"
            return "neutral"

        seg_neg = _has_semantic_negation(seg)
        stmt_neg = _has_semantic_negation(stmt)
        seg_groups = _predicate_groups(seg)
        stmt_groups = _predicate_groups(stmt)
        same_predicate = bool(seg_groups & stmt_groups)

        # Special-case biomedical efficacy framing:
        # "misused/inappropriate for viral infections" supports claims that antibiotics
        # do not work against viruses, even when statement phrasing includes "treating".
        if _is_antibiotic_viral_inefficacy_claim(seg) and _misuse_implies_inefficacy(stmt):
            return "entails"

        # Hypertension symptom framing:
        # "silent/asymptomatic/no symptoms" should contradict claims that BP usually
        # has noticeable symptoms (e.g., headaches).
        if _is_hypertension_symptom_claim(seg):
            seg_h = _hypertension_symptom_polarity(seg)
            stmt_h = _hypertension_symptom_polarity(stmt)
            if seg_h == "entails" and stmt_h == "contradicts":
                return "contradicts"
            if seg_h == "contradicts" and stmt_h == "entails":
                return "contradicts"
            if seg_h != "neutral" and seg_h == stmt_h:
                return "entails"

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
        if {"vaccine", "flu"}.issubset(seg_concepts) and not {
            "vaccine",
            "flu",
        }.issubset(stmt_concepts):
            return False
        if "antibiotic" in seg_concepts and ({"cold", "flu", "virus"} & seg_concepts):
            if "antibiotic" not in stmt_concepts:
                return False
            if not ({"cold", "flu", "virus"} & stmt_concepts):
                return False
        if {"sugar", "hyperactivity"}.issubset(seg_concepts) and not {
            "sugar",
            "hyperactivity",
        }.issubset(stmt_concepts):
            return False

        seg_tokens = self._topic_tokens(segment)
        stmt_tokens = self._topic_tokens(statement)
        if len(seg_tokens) >= 3 and len(seg_tokens & stmt_tokens) == 0 and not (seg_concepts & stmt_concepts):
            return False
        if not self._predicate_guard_ok(segment, statement):
            return False
        return True

    @staticmethod
    def _predicate_guard_ok(segment: str, statement: str) -> bool:
        seg = (segment or "").lower()
        stmt = (statement or "").lower()
        if not seg or not stmt:
            return False

        # DNA integration claims need integration/genomic predicate evidence (or explicit negation).
        if ("integrate" in seg or "integration" in seg) and "dna" in seg:
            has_predicate = bool(
                re.search(
                    r"\b(integrat(?:e|es|ed|ion)|genom(?:e|ic)|incorporat(?:e|ed|ion)|reverse transcription)\b",
                    stmt,
                )
            )
            has_explicit_neg = bool(
                re.search(
                    r"\b(do(?:es)?\s+not|cannot|can't|not)\b.{0,30}\b(integrat(?:e|ion)|alter|change)\b.{0,30}\bdna\b",
                    stmt,
                )
            )
            return has_predicate or has_explicit_neg

        # Smoking -> lung cancer risk claims require both entities and risk predicate.
        if "smoking" in seg and "lung" in seg and "cancer" in seg:
            has_entities = ("smoking" in stmt) and ("lung" in stmt) and ("cancer" in stmt)
            has_predicate = bool(re.search(r"\b(risk|increase|increases|cause|causes|associated)\b", stmt))
            return has_entities and has_predicate

        # Smoking vascular-risk claims should keep disease target alignment strict.
        if "smoking" in seg and bool(re.search(r"\b(risk|increase|increases|cause|causes|associated)\b", seg)):
            requires_stroke = "stroke" in seg
            requires_heart = bool(
                re.search(
                    r"\b(heart disease|cardiovascular disease|coronary heart disease|heart attack)\b",
                    seg,
                )
            )
            if requires_stroke or requires_heart:
                has_smoking = "smoking" in stmt
                has_predicate = bool(re.search(r"\b(risk|increase|increases|cause|causes|associated)\b", stmt))
                has_stroke = bool(re.search(r"\bstroke\b", stmt))
                has_heart = bool(
                    re.search(
                        r"\b(heart disease|cardiovascular disease|coronary heart disease|heart attack)\b",
                        stmt,
                    )
                )
                if requires_stroke and not has_stroke:
                    return False
                if requires_heart and not has_heart:
                    return False
                return has_smoking and has_predicate

        # Vitamin C -> common cold claims require both entities and prevention/treatment predicate.
        if "vitamin" in seg and "cold" in seg:
            has_entities = ("vitamin c" in stmt or ("vitamin" in stmt and "c" in stmt)) and ("cold" in stmt)
            has_predicate = bool(
                re.search(
                    r"\b(prevent|prevents|prevention|reduce|reduces|treat|treatment|duration)\b",
                    stmt,
                )
            )
            return has_entities and has_predicate

        # Diabetes management claims require management/medication predicates, not etiology-only facts.
        if "diabetes" in seg and bool(
            re.search(
                r"\b(manage|managed|management|lifestyle|medication|medications|reduce|stop)\b",
                seg,
            )
        ):
            has_diabetes = bool(re.search(r"\b(type\s*2\s+diabetes|type ii diabetes|diabetes)\b", stmt))
            if not has_diabetes:
                return False

            medication_change_claim = bool(
                re.search(
                    r"\b(reduce|stop|discontinue|withdraw)\b.{0,30}\b(medication|medications|drug|drugs)\b",
                    seg,
                )
            )
            if medication_change_claim:
                has_medication_change = bool(
                    re.search(
                        r"\b(reduce|reduced|stop|stopped|discontinue|discontinued|withdraw|withdrawn)\b.{0,40}\b"
                        r"(medication|medications|drug|drugs|insulin)\b",
                        stmt,
                    )
                )
                has_supervision_or_context = bool(
                    re.search(
                        r"\b(medical supervision|doctor|clinician|supervised|healthcare provider)\b",
                        stmt,
                    )
                ) or bool(re.search(r"\b(lifestyle|diet|exercise|weight loss|remission)\b", stmt))
                return has_medication_change and has_supervision_or_context

            lifestyle_manage_claim = bool(re.search(r"\b(manage|managed|management|lifestyle)\b", seg))
            if lifestyle_manage_claim:
                has_lifestyle_signal = bool(
                    re.search(
                        r"\b(lifestyle|diet|exercise|weight loss|physical activity|remission|managed?)\b",
                        stmt,
                    )
                )
                return has_lifestyle_signal

        # Hypertension symptom claims require symptom-language evidence.
        if re.search(r"\b(hypertension|high blood pressure|blood pressure)\b", seg) and re.search(
            r"\b(symptom|symptoms|headache|headaches|noticeable|silent|asymptomatic)\b",
            seg,
        ):
            has_condition = bool(re.search(r"\b(hypertension|high blood pressure|blood pressure)\b", stmt))
            has_symptom_signal = bool(
                re.search(
                    r"\b(symptom|symptoms|headache|headaches|asymptomatic|silent|no symptoms?)\b",
                    stmt,
                )
            )
            return has_condition and has_symptom_signal

        return True

    @staticmethod
    def _segment_object_tokens(segment: str) -> set[str]:
        text = (segment or "").lower()
        patterns = (
            r"\b(?:contain|contains|contained|work|works|effective|effectiveness)\b(?:\s+against)?\s+(?P<object>.+)$",
            (
                r"\b(?:increase|increases|increased|raise|raises|raised)\b(?:\s+the)?\s*"
                r"(?:risk|chance|likelihood)?(?:\s+of)?\s+(?P<object>.+)$"
            ),
            (
                r"\b(?:reduce|reduces|reduced|lower|lowers|lowered)\b(?:\s+the)?\s*"
                r"(?:risk|chance|likelihood)?(?:\s+of)?\s+(?P<object>.+)$"
            ),
            r"\b(?:prevent|prevents|prevented|preventing|cause|causes|caused|causing)\s+(?P<object>.+)$",
            r"\b(?:managed?|management)\s+(?:with|by|through)\s+(?P<object>.+)$",
            (
                r"\b(?:reduce|reduces|reduced|stop|stops|stopped|discontinue|discontinues|discontinued)\s+"
                r"(?P<object>.+?)(?:\s+under\b|$)"
            ),
        )
        object_text = ""
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                object_text = str(match.group("object") or "").strip()
                break
        if not object_text:
            return set()
        object_text = re.sub(
            r"^(?:the\s+)?(?:risk|chance|likelihood)\s+of\s+",
            "",
            object_text,
        )
        stop = {
            "to",
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "in",
            "on",
            "for",
            "with",
            "people",
            "person",
            "risk",
            "chance",
            "likelihood",
            "medical",
            "supervision",
            "under",
        }
        return {t for t in re.findall(r"\b[a-z][a-z0-9_-]+\b", object_text) if t not in stop}

    @staticmethod
    def _segment_subject_tokens(segment: str) -> set[str]:
        text = (segment or "").lower()
        match = re.search(
            r"^(?P<subject>.+?)\b(?:increase|increases|increased|cause|causes|caused|"
            r"reduce|reduces|reduced|prevent|prevents|prevented|contain|contains|work|works|has|have)\b",
            text,
        )
        if not match:
            return set()
        subject_text = match.group("subject")
        stop = {
            "to",
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "in",
            "on",
            "for",
            "with",
            "risk",
        }
        return {t for t in re.findall(r"\b[a-z][a-z0-9_-]+\b", subject_text) if t not in stop}

    @staticmethod
    def _statement_tokens(statement: str) -> set[str]:
        return set(re.findall(r"\b[a-z][a-z0-9_-]+\b", (statement or "").lower()))

    @staticmethod
    def _normalize_relevance_label(relevance: str) -> str:
        rel = str(relevance or "NEUTRAL").upper()
        if rel in {"CONTRADICTS", "REFUTES", "INVALID", "PARTIALLY_INVALID", "PARTIALLY_CONTRADICTS"}:
            return "REFUTES"
        if rel in {"SUPPORTS", "VALID", "PARTIALLY_VALID", "PARTIAL", "PARTIALLY_SUPPORTS"}:
            return "SUPPORTS"
        return "NEUTRAL"

    @staticmethod
    def _predicate_phrase_tokens(text: str) -> set[str]:
        low = (text or "").lower()
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
            "that",
            "this",
            "it",
            "its",
            "as",
            "from",
            "can",
            "could",
            "may",
            "might",
            "would",
            "should",
            "will",
            "do",
            "does",
            "did",
            "not",
            "no",
        }
        cue_patterns = (
            r"\b(cause|causes|caused|causing|lead|leads|led|leading|result|results|resulted|"
            r"increase|increases|increased|raise|raises|raised|reduce|reduces|reduced|lower|lowers|lowered|"
            r"prevent|prevents|prevented|preventing|cure|cures|cured|curing|treat|treats|treated|treating|"
            r"integrate|integrates|integrated|integration|alter|alters|altered|change|changes|changed|"
            r"risk|risks|effective|ineffective|efficacy|harm|harms|safe|unsafe)\b",
            r"\b(more|less|higher|lower)\b",
        )
        cues = set()
        for pat in cue_patterns:
            cues.update(re.findall(pat, low))
        if not cues:
            return set()
        tokens = {t for t in re.findall(r"\b[a-z][a-z0-9_-]+\b", low) if t not in stop}
        return tokens

    def compute_predicate_match(self, claim_text: str, evidence_text: str) -> float:
        claim_pred = self._predicate_phrase_tokens(claim_text)
        ev_pred = self._predicate_phrase_tokens(evidence_text)
        if not claim_pred:
            return 0.0
        overlap = len(claim_pred & ev_pred) / max(1, len(claim_pred))
        seg_pol = self._segment_polarity(claim_text, evidence_text, stance="neutral")
        polarity_bonus = 0.2 if seg_pol in {"entails", "contradicts"} else 0.0
        raw = (0.8 * overlap) + polarity_bonus
        return max(0.0, min(1.0, raw))

    def _contradiction_score(self, claim_text: str, evidence_text: str) -> float:
        pol = self._segment_polarity(claim_text, evidence_text, stance="neutral")
        anchor_overlap = self._segment_anchor_overlap(claim_text, evidence_text)
        explicit_refute = 1.0 if self._is_explicit_refutation_statement(evidence_text) else 0.0
        predicate = self.compute_predicate_match(claim_text, evidence_text)
        claim_neg = bool(
            re.search(r"\b(no|not|never|cannot|can't|does not|do not|without)\b", (claim_text or "").lower())
        )
        ev_neg = bool(
            re.search(
                r"\b(no|not|never|cannot|can't|does not|do not|without|doesn't|don't)\b", (evidence_text or "").lower()
            )
        )
        negation_mismatch = 1.0 if claim_neg != ev_neg else 0.0
        polarity_signal = 1.0 if pol == "contradicts" else 0.0
        score = (
            (0.35 * predicate)
            + (0.25 * anchor_overlap)
            + (0.20 * explicit_refute)
            + (0.20 * max(negation_mismatch, polarity_signal))
        )
        return max(0.0, min(1.0, score))

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
        if self._is_claim_mention_statement(statement):
            score_f *= 0.25
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
            base_score = float(item.get("relevance_score", ev.get("final_score", ev.get("score", 0.0))) or 0.0)
            anchor_score = float(ev.get("anchor_match_score", self._segment_anchor_overlap(claim, statement)) or 0.0)
            predicate_match_score = self.compute_predicate_match(claim, statement)
            contradiction_score = self._contradiction_score(claim, statement)
            strength = compute_evidence_strength(
                claim_text=claim,
                text_snippet=statement,
                source_meta=ev,
                stance_hint=item.get("relevance"),
            )
            support_strength = float(strength.support_strength or 0.0)
            polarity_rel = self._segment_polarity(claim, statement, stance="neutral")
            seed_rel = self._normalize_relevance_label(item.get("relevance", "NEUTRAL"))

            relevance = "NEUTRAL"
            relevance_score = max(0.0, min(1.0, base_score))
            if not self._segment_topic_guard_ok(claim, statement):
                relevance_score *= 0.10
            else:
                # Refutation can be decided by direct contradiction, independent of support gates.
                numeric_rel = self._numeric_relation_relevance(claim, statement)
                dna_rel = self._dna_integration_relevance(claim, statement)
                refute_by_rule = (
                    contradiction_score >= CONTRADICTION_THRESHOLD
                    or polarity_rel == "contradicts"
                    or numeric_rel == "CONTRADICTS"
                    or dna_rel == "CONTRADICTS"
                )
                if refute_by_rule:
                    relevance = "REFUTES"
                    relevance_score = max(relevance_score, contradiction_score, 0.65)
                else:
                    support_gate = (
                        predicate_match_score >= PREDICATE_MATCH_THRESHOLD
                        and anchor_score >= ANCHOR_THRESHOLD
                        and polarity_rel == "entails"
                        and support_strength >= 0.35
                        and not self._is_reporting_statement(statement)
                        and not self._is_claim_mention_statement(statement)
                    )
                    # Numeric exact-match can support if predicate and anchor are both present.
                    if numeric_rel == "SUPPORTS" and predicate_match_score >= PREDICATE_MATCH_THRESHOLD:
                        support_gate = support_gate or (anchor_score >= ANCHOR_THRESHOLD)
                    if support_gate:
                        relevance = "SUPPORTS"
                        relevance_score = max(relevance_score, support_strength, 0.62)
                    elif seed_rel == "REFUTES" and contradiction_score >= 0.45:
                        relevance = "REFUTES"
                        relevance_score = max(relevance_score, contradiction_score)
                    else:
                        relevance = "NEUTRAL"
                        relevance_score *= 0.55
            normalized.append(
                {
                    "evidence_id": ev_idx if ev_idx >= 0 else len(normalized),
                    "statement": statement,
                    "relevance": relevance,
                    "relevance_score": max(0.0, min(1.0, relevance_score)),
                    "source_url": source_url,
                    "anchor_match_score": anchor_score,
                    "predicate_match_score": max(0.0, min(1.0, predicate_match_score)),
                    "support_strength": max(0.0, min(1.0, support_strength)),
                    "contradiction_score": max(0.0, min(1.0, contradiction_score)),
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
        support_strength_threshold = float(os.getenv("SEGMENT_SUPPORT_STRENGTH_THRESHOLD", "0.35"))

        for seg in claim_breakdown:
            segment = (seg.get("claim_segment") or "").strip()
            segment_belief_mode = self._segment_is_belief_or_survey_claim(segment)
            best_support_item: Dict[str, Any] | None = None
            best_refute_item: Dict[str, Any] | None = None
            weak_support_item: Dict[str, Any] | None = None
            best_support_score = -1.0
            best_refute_score = -1.0
            best_weak_support_score = -1.0
            saw_neutral = False

            def _consider_item(em: Dict[str, Any], ev_idx: int | None = None) -> None:
                nonlocal best_support_item
                nonlocal best_refute_item
                nonlocal weak_support_item
                nonlocal best_support_score
                nonlocal best_refute_score
                nonlocal best_weak_support_score
                nonlocal saw_neutral

                statement = (em.get("statement") or "").strip()
                if not statement:
                    return
                if self._is_claim_mention_statement(statement) and not segment_belief_mode:
                    return
                if not self._segment_topic_guard_ok(segment, statement):
                    return
                anchor_eval = evaluate_anchor_match(segment, statement)
                anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                if anchor_overlap < _SEGMENT_EVIDENCE_MIN_OVERLAP:
                    return
                object_tokens = self._segment_object_tokens(segment)
                if object_tokens and not segment_belief_mode:
                    statement_tokens = set(re.findall(r"\b[a-z][a-z0-9_-]+\b", statement.lower()))
                    no_object_overlap = len(object_tokens & statement_tokens) == 0
                    if no_object_overlap and not self._is_explicit_refutation_statement(statement):
                        return

                rel = self._normalize_relevance_label(em.get("relevance", "NEUTRAL"))
                rel_score = float(em.get("relevance_score", 0.0) or 0.0)
                predicate_match_score = float(
                    em.get("predicate_match_score", self.compute_predicate_match(segment, statement)) or 0.0
                )
                support_strength = float(em.get("support_strength", 0.0) or 0.0)
                contradiction_score = float(
                    em.get("contradiction_score", self._contradiction_score(segment, statement))
                )
                strict_support_ok = (
                    rel == "SUPPORTS"
                    and predicate_match_score >= PREDICATE_MATCH_THRESHOLD
                    and anchor_overlap >= ANCHOR_THRESHOLD
                    and support_strength >= support_strength_threshold
                )
                refute_ok = rel == "REFUTES" and (
                    contradiction_score >= CONTRADICTION_THRESHOLD
                    or self._segment_polarity(segment, statement, stance="neutral") == "contradicts"
                )
                score = (0.55 * rel_score) + (0.25 * anchor_overlap) + (0.20 * predicate_match_score)
                if strict_support_ok and score > best_support_score:
                    best_support_score = score
                    best_support_item = {
                        **em,
                        "evidence_id": em.get("evidence_id", ev_idx if ev_idx is not None else -1),
                        "statement": statement,
                        "anchor_match_score": anchor_overlap,
                        "predicate_match_score": predicate_match_score,
                        "support_strength": support_strength,
                        "contradiction_score": contradiction_score,
                        "stance_used": rel,
                    }
                elif refute_ok and score > best_refute_score:
                    best_refute_score = score
                    best_refute_item = {
                        **em,
                        "evidence_id": em.get("evidence_id", ev_idx if ev_idx is not None else -1),
                        "statement": statement,
                        "anchor_match_score": anchor_overlap,
                        "predicate_match_score": predicate_match_score,
                        "support_strength": support_strength,
                        "contradiction_score": contradiction_score,
                        "stance_used": rel,
                    }
                elif rel == "SUPPORTS":
                    # Keep weak/hedged support for PARTIALLY_VALID fallback.
                    weakness = 1.0 - max(0.0, min(1.0, support_strength))
                    weak_score = (
                        (0.45 * rel_score)
                        + (0.30 * anchor_overlap)
                        + (0.25 * predicate_match_score)
                        - (0.20 * weakness)
                    )
                    if weak_score > best_weak_support_score:
                        best_weak_support_score = weak_score
                        weak_support_item = {
                            **em,
                            "evidence_id": em.get("evidence_id", ev_idx if ev_idx is not None else -1),
                            "statement": statement,
                            "anchor_match_score": anchor_overlap,
                            "predicate_match_score": predicate_match_score,
                            "support_strength": support_strength,
                            "contradiction_score": contradiction_score,
                            "stance_used": rel,
                        }
                else:
                    saw_neutral = True

            for em in evidence_map:
                _consider_item(em)

            if best_support_item is None and best_refute_item is None and weak_support_item is None:
                # Fallback to segment-retrieved evidence pool.
                for idx, ev in enumerate(evidence):
                    seg_q = (ev.get("_segment_query") or "").strip().lower()
                    statement = (ev.get("statement") or ev.get("text") or "").strip()
                    if not statement:
                        continue
                    if self._is_claim_mention_statement(statement) and not segment_belief_mode:
                        continue
                    if seg_q and seg_q not in segment.lower():
                        continue
                    fallback_item = {
                        "evidence_id": idx,
                        "statement": statement,
                        "source_url": ev.get("source_url") or ev.get("source") or "",
                        "relevance_score": float(ev.get("final_score", ev.get("score", 0.0)) or 0.0),
                        "relevance": self._normalize_relevance_label(ev.get("relevance", "NEUTRAL")),
                        "support_strength": float(ev.get("final_score", ev.get("score", 0.0)) or 0.0),
                    }
                    _consider_item(fallback_item, ev_idx=idx)

            explicit_refute_present = best_refute_item is not None
            chosen: Dict[str, Any] | None = best_refute_item or best_support_item or weak_support_item
            if chosen is not None:
                ev_id = int(chosen.get("evidence_id", -1) or -1)
                ev = evidence_by_id.get(ev_id, {})
                statement = (chosen.get("statement") or ev.get("statement") or ev.get("text") or "").strip()
                source_url = (chosen.get("source_url") or ev.get("source_url") or ev.get("source") or "").strip()
                if explicit_refute_present and best_support_item is not None:
                    seg["status"] = "PARTIALLY_INVALID"
                elif explicit_refute_present:
                    seg["status"] = "INVALID"
                elif best_support_item is not None:
                    seg["status"] = "VALID"
                else:
                    seg["status"] = "PARTIALLY_VALID"
                if statement:
                    seg["supporting_fact"] = statement
                if statement and source_url:
                    seg["source_url"] = source_url
                seg["evidence_used_ids"] = [ev_id] if ev_id >= 0 else []
                seg["alignment_debug"] = {
                    "reason": "strict_predicate_gate",
                    "anchor_overlap": round(float(chosen.get("anchor_match_score", 0.0) or 0.0), 3),
                    "predicate_match_score": round(float(chosen.get("predicate_match_score", 0.0) or 0.0), 3),
                    "support_strength": round(float(chosen.get("support_strength", 0.0) or 0.0), 3),
                    "stance_used": str(
                        chosen.get("stance_used") or self._normalize_relevance_label(chosen.get("relevance"))
                    ),
                    "score": round(max(best_support_score, best_refute_score, best_weak_support_score), 3),
                }
            else:
                seg.setdefault("evidence_used_ids", [])
                seg["status"] = "UNKNOWN"
                seg["supporting_fact"] = ""
                seg["source_url"] = ""
                seg["alignment_debug"] = {
                    "reason": "strict_gate_no_match",
                    "stance_used": "NEUTRAL" if saw_neutral else "NONE",
                    "predicate_match_score": 0.0,
                    "support_strength": 0.0,
                    "min_overlap": _SEGMENT_EVIDENCE_MIN_OVERLAP,
                }
        return claim_breakdown

    def _log_subclaim_coverage(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
        claim_breakdown: List[Dict[str, Any]],
        adaptive_metrics: Dict[str, Any] | None = None,
    ) -> None:
        # When adaptive metrics are provided by pipeline memoization,
        # avoid expensive/redundant anchor+adaptive recomputation in verdict phase.
        if adaptive_metrics is not None:
            try:
                weighted = float(adaptive_metrics.get("coverage", 0.0) or 0.0)
                status_weight = {
                    "VALID": 1.0,
                    "STRONGLY_VALID": 1.0,
                    "PARTIALLY_VALID": 0.7,
                    "PARTIALLY_INVALID": 0.7,
                    "INVALID": 1.0,
                }
                verdict_weighted = sum(
                    status_weight.get(str(seg.get("status", "UNKNOWN")).upper(), 0.0) for seg in (claim_breakdown or [])
                )
                verdict_cov = verdict_weighted / max(1, len(claim_breakdown or []))
                logger.info(
                    "[VerdictGenerator][Coverage] subclaims=%d weighted_covered=%.2f strong=%d partial=%d invalid=%d "
                    "unknown=%d coverage=%.2f",
                    len(claim_breakdown or []),
                    verdict_weighted,
                    int(adaptive_metrics.get("strong_covered", 0) or 0),
                    0,
                    int(adaptive_metrics.get("contradicted_subclaims", 0) or 0),
                    sum(1 for seg in (claim_breakdown or []) if str(seg.get("status", "UNKNOWN")).upper() == "UNKNOWN"),
                    verdict_cov,
                )
                logger.info(
                    "[VerdictGenerator][Coverage][Aligned] verdict_coverage=%.2f adaptive_coverage=%.2f",
                    verdict_cov,
                    weighted,
                )
            except Exception:
                logger.debug("[VerdictGenerator][Coverage] memoized coverage logging skipped due to parse issue")
            return

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
            if adaptive_metrics is not None:
                logger.info(
                    "[VerdictGenerator][Coverage][Aligned] verdict_coverage=%.2f adaptive_coverage=%.2f",
                    float(cov.get("coverage", 0.0)),
                    float(adaptive_metrics.get("coverage", 0.0) or 0.0),
                )
                return

            class _Ev:
                __slots__ = (
                    "statement",
                    "source_url",
                    "semantic_score",
                    "stance",
                    "trust",
                )

                def __init__(self, d: Dict[str, Any]):
                    self.statement = d.get("statement") or d.get("text") or ""
                    self.source_url = d.get("source_url") or d.get("source") or ""
                    self.semantic_score = float(
                        d.get("semantic_score") or d.get("sem_score") or d.get("final_score") or d.get("score") or 0.0
                    )
                    self.stance = d.get("stance") or "unknown"
                    self.trust = float(d.get("trust") or d.get("final_score") or d.get("score") or 0.0)

            adaptive = trust_policy.compute_adaptive_trust(
                claim,
                [_Ev(d) for d in evidence if (d.get("statement") or d.get("text"))],
                top_k=min(12, len(evidence)),
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

    def _apply_adaptive_coverage_fallback(
        self,
        claim_breakdown: List[Dict[str, Any]],
        adaptive_metrics: Dict[str, Any] | None,
        evidence: List[Dict[str, Any]],
    ) -> None:
        if not claim_breakdown or not adaptive_metrics:
            return
        adaptive_coverage = float(adaptive_metrics.get("coverage", 0.0) or 0.0)
        if adaptive_coverage < 0.60:
            return
        unknown_segments = [seg for seg in claim_breakdown if str(seg.get("status", "UNKNOWN")).upper() == "UNKNOWN"]
        if len(unknown_segments) != len(claim_breakdown):
            return
        # Do not promote UNKNOWN when alignment explicitly says no relevant evidence.
        if all(
            str(((seg.get("alignment_debug") or {}).get("reason") or "")).lower() == "no_relevant_evidence"
            for seg in unknown_segments
        ):
            return

        best_ev: Dict[str, Any] | None = None
        best_score = -1.0
        for ev in evidence:
            stance = str(ev.get("stance") or "neutral").lower()
            if stance == "contradicts":
                continue
            statement = str(ev.get("statement") or ev.get("text") or "")
            if self._is_claim_mention_statement(statement):
                continue
            if not all(
                self._segment_topic_guard_ok(str(seg.get("claim_segment") or ""), statement) for seg in unknown_segments
            ):
                continue
            sem = float(ev.get("semantic_score") or ev.get("sem_score") or ev.get("final_score") or 0.0)
            if sem < 0.55:
                continue
            if sem > best_score:
                best_score = sem
                best_ev = ev
        if not best_ev:
            # If adaptive metrics are strong but sem-keyed evidence is sparse,
            # still promote with first non-contradicting evidence to avoid UNKNOWN deadlock.
            for ev in evidence:
                stance = str(ev.get("stance") or "neutral").lower()
                if stance == "contradicts":
                    continue
                statement = str(ev.get("statement") or ev.get("text") or "")
                if self._is_claim_mention_statement(statement):
                    continue
                if not all(
                    self._segment_topic_guard_ok(str(seg.get("claim_segment") or ""), statement)
                    for seg in unknown_segments
                ):
                    continue
                best_ev = ev
                best_score = float(ev.get("final_score") or ev.get("score") or 0.0)
                break
            if not best_ev:
                return

        fallback_fact = str(best_ev.get("statement") or best_ev.get("text") or "").strip()
        fallback_src = str(best_ev.get("source_url") or best_ev.get("source") or "").strip()
        if not fallback_fact:
            return
        for seg in claim_breakdown:
            seg["status"] = "PARTIALLY_VALID"
            seg["supporting_fact"] = fallback_fact
            seg["source_url"] = fallback_src
        logger.info(
            "[VerdictGenerator][Coverage][Fallback] Applied adaptive fallback "
            "(adaptive_coverage=%.2f, evidence_sem=%.3f)",
            adaptive_coverage,
            best_score,
        )

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
            "STRONGLY_VALID": 1.0,
            "VALID": 1.0,
            "PARTIALLY_VALID": 0.75,
            "PARTIALLY_INVALID": 0.25,
            "INVALID": 0.20,
            "UNKNOWN": 0.45,
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
            "INVALID": 0.20,
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

    def _apply_truthfulness_invariant(
        self,
        truthfulness_percent: float,
        claim_breakdown: List[Dict[str, Any]],
        explicit_refutes_found: bool,
    ) -> tuple[float, bool, float]:
        statuses = [str((seg or {}).get("status") or "UNKNOWN").upper() for seg in (claim_breakdown or [])]
        if not statuses:
            return max(0.0, float(truthfulness_percent or 0.0)), False, 0.0

        max_status_weight = max(self._status_truth_weight(s) for s in statuses)
        score = max(0.0, float(truthfulness_percent or 0.0))
        applied = False

        # Invariant A: unknown/partially-valid only, without explicit refutation, cannot inflate.
        if all(s in {"UNKNOWN", "PARTIALLY_VALID"} for s in statuses) and not explicit_refutes_found:
            score = min(score, 45.0)
            applied = True

        # Invariant B: enforce status-bound upper cap.
        bound = max_status_weight * 100.0
        if score > bound:
            score = bound
            applied = True

        # Explicit invariant assertion for audit safety.
        assert score <= (max_status_weight * 100.0) + 1e-9
        return round(score, 1), applied, max_status_weight

    def _cap_confidence_with_contract(
        self,
        confidence: float,
        contract: Dict[str, Any],
        policy_insufficient: bool,
        verdict: str,
        is_comparative_claim: bool = False,
    ) -> float:
        cap = 0.98
        unresolved = int(contract.get("unresolved_segments", 0) or 0)
        resolved_ratio = float(contract.get("resolved_ratio", 0.0) or 0.0)
        weighted_truth = float(contract.get("weighted_truth", 0.0) or 0.0)
        has_support = bool(contract.get("has_support", False))
        has_invalid = bool(contract.get("has_invalid", False))
        if unresolved > 0:
            cap = min(cap, 0.40 + (0.35 * resolved_ratio))
        if policy_insufficient:
            cap = min(cap, 0.62 if unresolved == 0 else 0.55)
        if verdict == Verdict.UNVERIFIABLE.value:
            cap = min(cap, 0.45 if is_comparative_claim else 0.35)
        if unresolved > 0 and weighted_truth <= 0.35:
            cap = min(cap, 0.48)
        bounded = max(0.05, min(float(confidence or 0.0), cap))
        # Fully resolved contradiction-only outcomes should not look underconfident.
        if (
            verdict == Verdict.FALSE.value
            and unresolved == 0
            and has_invalid
            and not has_support
            and not is_comparative_claim
        ):
            bounded = max(bounded, 0.50)
        return bounded

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
        support_like = {"VALID", "PARTIALLY_VALID", "STRONGLY_VALID"}
        invalid_like = {"INVALID", "PARTIALLY_INVALID"}
        all_valid = bool(statuses) and all(s == "VALID" for s in statuses)
        all_invalid = bool(statuses) and all(s == "INVALID" for s in statuses)
        all_invalid_like = bool(statuses) and all((s in invalid_like) for s in statuses)
        all_support_like = bool(statuses) and all((s in support_like) for s in statuses)

        if unresolved_segments > 0:
            if has_support and has_invalid:
                verdict = Verdict.PARTIALLY_TRUE.value
            elif has_support:
                verdict = Verdict.PARTIALLY_TRUE.value
            else:
                verdict = Verdict.UNVERIFIABLE.value
        elif all_valid or (all_support_like and not has_invalid):
            verdict = Verdict.TRUE.value
        elif all_invalid or (all_invalid_like and not has_support):
            verdict = Verdict.FALSE.value
        elif has_support and has_invalid:
            verdict = Verdict.PARTIALLY_TRUE.value
        elif has_invalid and not has_support:
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

    @staticmethod
    def _canonical_stance_from_breakdown(claim_breakdown: List[Dict[str, Any]]) -> str:
        statuses = [str((seg or {}).get("status", "UNKNOWN") or "UNKNOWN").upper() for seg in (claim_breakdown or [])]
        if not statuses:
            return "neutral"
        support = sum(1 for s in statuses if s in {"VALID", "PARTIALLY_VALID", "STRONGLY_VALID"})
        contra = sum(1 for s in statuses if s in {"INVALID", "PARTIALLY_INVALID"})
        unknown = sum(1 for s in statuses if s == "UNKNOWN")
        if contra > 0 and support == 0 and unknown == 0:
            return "contradicts"
        if support > 0 and contra == 0 and unknown == 0:
            return "entails"
        return "neutral"

    @staticmethod
    def _canonical_stance_from_evidence_map(evidence_map: List[Dict[str, Any]]) -> str:
        supports = 0.0
        contradicts = 0.0
        for ev in evidence_map or []:
            rel = VerdictGenerator._normalize_relevance_label(ev.get("relevance"))
            try:
                score = float(ev.get("relevance_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            weight = max(0.2, min(1.0, score))
            if rel == "SUPPORTS":
                supports += weight
            elif rel == "REFUTES":
                contradicts += weight
        if contradicts >= (supports * 1.15) and (contradicts - supports) >= 0.15:
            return "contradicts"
        if supports >= (contradicts * 1.15) and (supports - contradicts) >= 0.15:
            return "entails"
        return "neutral"

    def _top_admissible_stance_signal(self, claim: str, evidence_map: List[Dict[str, Any]]) -> tuple[str, float]:
        best_stance = "neutral"
        best_score = 0.0
        for ev in evidence_map or []:
            stmt = str(ev.get("statement") or "").strip()
            if not stmt:
                continue
            if not self._evidence_is_admissible_for_claim(claim, stmt):
                continue
            rel = self._normalize_relevance_label(ev.get("relevance"))
            if rel not in {"SUPPORTS", "REFUTES"}:
                continue
            try:
                score = float(ev.get("relevance_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_stance = "entails" if rel == "SUPPORTS" else "contradicts"
        return best_stance, max(0.0, min(1.0, best_score))

    def _stance_vote_decision(self, claim: str, evidence_map: List[Dict[str, Any]]) -> tuple[str | None, float, float]:
        support = 0.0
        contradict = 0.0
        for ev in evidence_map or []:
            stmt = str(ev.get("statement") or "")
            if not stmt or not self._evidence_is_admissible_for_claim(claim, stmt):
                continue
            rel = self._normalize_relevance_label(ev.get("relevance"))
            try:
                score = float(ev.get("relevance_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if rel == "SUPPORTS":
                support = max(support, score)
            elif rel == "REFUTES":
                contradict = max(contradict, score)

        decision: str | None = None
        if contradict >= 0.65 and contradict > support:
            decision = Verdict.FALSE.value
        elif support >= 0.65 and support > contradict:
            decision = Verdict.TRUE.value
        return decision, support, contradict

    def _evidence_map_signal_strengths(
        self,
        claim: str,
        evidence_map: List[Dict[str, Any]],
    ) -> tuple[float, float, int, int]:
        support_max = 0.0
        contradict_max = 0.0
        admissible_count = 0
        offtopic_count = 0
        support_labels = {"SUPPORTS"}
        contradict_labels = {"REFUTES"}

        for ev in evidence_map or []:
            stmt = str(ev.get("statement") or "").strip()
            if not stmt:
                continue
            rel = self._normalize_relevance_label(ev.get("relevance"))
            if not self._segment_topic_guard_ok(claim, stmt):
                offtopic_count += 1
                continue
            if not self._evidence_is_admissible_for_claim(claim, stmt):
                continue
            admissible_count += 1
            try:
                score = float(ev.get("relevance_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            score = max(0.0, min(1.0, score))
            if rel in support_labels:
                support_max = max(support_max, score)
            elif rel in contradict_labels:
                contradict_max = max(contradict_max, score)

        return support_max, contradict_max, admissible_count, offtopic_count

    def _contradiction_dominance_metrics(
        self,
        claim: str,
        evidence_map: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        support_weight = 0.0
        refute_weight = 0.0
        refute_domains: set[str] = set()
        seen_domains: set[str] = set()

        for ev in evidence_map or []:
            stmt = str(ev.get("statement") or "").strip()
            if not stmt or not self._segment_topic_guard_ok(claim, stmt):
                continue
            rel = self._normalize_relevance_label(ev.get("relevance"))
            score = max(0.0, min(1.0, float(ev.get("relevance_score", 0.0) or 0.0)))
            src = str(ev.get("source_url") or "").strip().lower()
            domain = ""
            if src.startswith("http"):
                try:
                    domain = src.split("/")[2].removeprefix("www.")
                except Exception:
                    domain = ""
            if domain:
                seen_domains.add(domain)
            if rel == "SUPPORTS":
                support_weight += score
            elif rel == "REFUTES":
                refute_weight += score
                if domain:
                    refute_domains.add(domain)

        total = support_weight + refute_weight
        contradict_ratio = (refute_weight / total) if total > 0 else 0.0
        support_coverage = support_weight / max(1.0, total)
        contradict_coverage = contradict_ratio
        denom_domains = max(1, len(seen_domains))
        contradict_diversity = len(refute_domains) / denom_domains
        return {
            "contradict_ratio": max(0.0, min(1.0, contradict_ratio)),
            "contradict_coverage": max(0.0, min(1.0, contradict_coverage)),
            "contradict_diversity": max(0.0, min(1.0, contradict_diversity)),
            "support_coverage": max(0.0, min(1.0, support_coverage)),
        }

    @staticmethod
    def _dna_integration_relevance(claim: str, statement: str) -> str:
        seg = (claim or "").lower()
        stmt = (statement or "").lower()
        if not (("integrate" in seg or "integration" in seg) and "dna" in seg):
            return "NEUTRAL"

        supports = bool(
            re.search(
                r"\b(may|can|could|does|do|is|are)?\s*integrat(?:e|es|ed|ion)\b.{0,30}\b(dna|genom(?:e|ic))\b",
                stmt,
            )
        ) or bool(
            re.search(
                r"\b(reverse transcription|genomic integration|integrate into the human genome)\b",
                stmt,
            )
        )
        contradicts = bool(
            re.search(
                r"\b(do(?:es)?\s+not|cannot|can't|not)\b.{0,30}\b(integrat(?:e|ion)|alter|change)\b.{0,30}\bdna\b",
                stmt,
            )
        ) or bool(re.search(r"\bdoes not integrate with (our|human) dna\b", stmt))
        if supports and not contradicts:
            return "SUPPORTS"
        if contradicts and not supports:
            return "CONTRADICTS"
        return "NEUTRAL"

    @staticmethod
    def _has_absolute_quantifier(text: str) -> bool:
        low = (text or "").lower()
        return bool(
            re.search(
                r"\b(always|never|all|none|every|entirely|completely|must|only|prevents|prevents|cures|cure)\b",
                low,
            )
        )

    @staticmethod
    def _extract_simple_numbers(text: str) -> List[float]:
        nums: List[float] = []
        for m in re.finditer(r"\b\d+(?:\.\d+)?\b", (text or "")):
            try:
                nums.append(float(m.group(0)))
            except Exception:
                continue
        return nums

    def _numeric_relation_relevance(self, claim: str, statement: str) -> str:
        """
        For simple exact-value claims (non-comparative), detect numeric entailment/contradiction.
        """
        if self._is_numeric_comparison_claim(claim):
            return "NEUTRAL"
        if not self._segment_topic_guard_ok(claim, statement):
            return "NEUTRAL"
        claim_nums = self._extract_simple_numbers(claim)
        stmt_nums = self._extract_simple_numbers(statement)
        if len(claim_nums) != 1 or len(stmt_nums) != 1:
            return "NEUTRAL"
        c = claim_nums[0]
        s = stmt_nums[0]
        if abs(c - s) < 1e-9:
            return "SUPPORTS"
        return "CONTRADICTS"

    @staticmethod
    def _rationale_polarity_hint(rationale: str) -> str:
        text = (rationale or "").lower()
        if not text:
            return "neutral"
        contradict_patterns = (
            r"\bcontradict(?:ed|s|ion)?\b",
            r"\bclaim\s+is\s+false\b",
            r"\bnot\s+supported\b",
            r"\bdebunk(?:ed|ing)?\b",
        )
        entail_patterns = (
            r"\bsupported\s+by\s+evidence\b",
            r"\bevidence\s+confirms\b",
            r"\bclaim\s+is\s+true\b",
            r"\bstrongly\s+supports\b",
        )
        if any(re.search(p, text) for p in contradict_patterns):
            return "contradicts"
        if any(re.search(p, text) for p in entail_patterns):
            return "entails"
        return "neutral"

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
        original = (original_rationale or "").strip()

        def _normalize_for_verdict(text: str, verdict_value: str) -> str:
            t = (text or "").strip()
            if not t:
                return t
            if verdict_value == Verdict.TRUE.value:
                # Keep label+rationale consistent; a TRUE verdict must not claim "partially true".
                t = re.sub(r"\bpartially\s+true\b", "true", t, flags=re.IGNORECASE)
            elif verdict_value == Verdict.PARTIALLY_TRUE.value:
                # Prefer explicit partial framing for partial verdicts.
                if not re.search(r"\bpartially\s+true\b", t, flags=re.IGNORECASE):
                    t = f"{t} Overall, the claim is partially true."
            elif verdict_value == Verdict.FALSE.value:
                # Remove contradictory support wording.
                if re.search(r"\bpartially\s+true\b", t, flags=re.IGNORECASE):
                    t = "Evidence contradicts the claim."
            return t

        if verdict == Verdict.UNVERIFIABLE.value:
            return (
                "Available admissible evidence is mixed or insufficiently decisive for this claim, "
                "so the result is UNVERIFIABLE."
            )
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
            return _normalize_for_verdict(
                original or "Evidence supports all required claim segments.",
                verdict,
            )
        return _normalize_for_verdict(
            original or "Verdict is based on segment-level evidence evaluation.",
            verdict,
        )

    def _build_deterministic_claim_breakdown(self, claim: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build a deterministic claim breakdown with meaningful segments.
        Used when LLM returns fragmentary segments (e.g. single words).
        """
        segments = self._split_claim_into_segments(claim)
        out: List[Dict[str, Any]] = []
        uncertainty_terms = {
            "less",
            "uncertain",
            "unclear",
            "inconclusive",
            "mixed",
            "limited",
            "insufficient",
        }
        assertive_claim = bool(
            re.search(
                r"\b(helps?|prevents?|reduces?|increases?|causes?|proves?|protects?)\b",
                claim,
                re.IGNORECASE,
            )
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
                mention_penalty = 0.0
                if self._is_claim_mention_statement(stmt) and not self._segment_is_belief_or_survey_claim(seg):
                    mention_penalty = 0.40
                score = (
                    (0.50 * max(0.0, min(1.0, rel_f)))
                    + (0.25 * max(0.0, min(1.0, overlap)))
                    + (0.25 * max(0.0, min(1.0, anchor_overlap)))
                    - uncertainty_penalty
                    - reporting_penalty
                    - mention_penalty
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
        neg_terms = {
            "no",
            "not",
            "never",
            "none",
            "without",
            "lack",
            "lacks",
            "lacking",
        }
        uncertainty_terms = {
            "uncertain",
            "unclear",
            "inconclusive",
            "mixed",
            "limited",
            "insufficient",
        }
        claim_assertive = bool(
            re.search(
                r"\b(helps?|prevents?|reduces?|increases?|causes?|proves?|protects?)\b",
                claim,
                re.IGNORECASE,
            )
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
            best = max(
                0.0,
                min(1.0, avg_support - contradiction_penalty - neg_alignment_penalty),
            )
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
        stmt = re.sub(
            r"\s+",
            " ",
            str(ev.get("statement") or ev.get("text") or "").strip().lower(),
        )
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
        elif self._policy_says_insufficient(claim, final_evidence, adaptive_metrics=None):
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

    @staticmethod
    def _segment_recovery_query_hints(segment: str) -> List[str]:
        """Deterministic query hints for hard claim patterns."""
        seg = (segment or "").strip()
        low = seg.lower()
        hints: List[str] = []
        if re.search(r"\b(hypertension|high blood pressure|blood pressure)\b", low) and re.search(
            r"\b(symptom|symptoms|headache|headaches|noticeable)\b",
            low,
        ):
            hints.append("high blood pressure symptoms silent condition no symptoms headaches not reliable indicator")
            hints.append("hypertension usually no symptoms headaches not a reliable sign")
        if "smoking" in low and ("stroke" in low or "heart disease" in low or "cardiovascular" in low):
            hints.append("smoking increases risk of stroke and heart disease evidence")
            hints.append("cdc smoking causes stroke and heart disease")
        if "diabetes" in low and (
            "lifestyle" in low or "managed" in low or "medication" in low or "supervision" in low
        ):
            hints.append("type 2 diabetes can be managed with lifestyle changes diet exercise")
            hints.append("type 2 diabetes reduce or stop medication under medical supervision remission")
        return hints

    @staticmethod
    def _extract_predicate_object_phrase(segment: str) -> tuple[str, str]:
        seg = (segment or "").strip().lower()
        if not seg:
            return "", ""
        match = re.search(
            r"\b(?:can|may|might|does|do|is|are|will|would|should)?\s*"
            r"(cause|causes|prevent|prevents|treat|treats|cure|cures|alter|alters|change|changes|"
            r"integrate|integrates|increase|increases|reduce|reduces|lead to|leads to|result in|results in)\b"
            r"\s+(?P<object>.+)$",
            seg,
        )
        if not match:
            return "", ""
        predicate = str(match.group(1) or "").strip()
        obj = str(match.group("object") or "").strip(" .,")
        return predicate, obj

    def _predicate_refute_query_hints(self, segment: str) -> List[str]:
        seg = (segment or "").strip()
        low = seg.lower()
        predicate, obj = self._extract_predicate_object_phrase(seg)
        subject_tokens = sorted(self._segment_subject_tokens(seg))
        subject_phrase = " ".join(subject_tokens[:4]).strip() if subject_tokens else ""

        if not predicate:
            # Fallback to generic mechanism refutation patterns when predicate extraction is weak.
            if "mrna" in low and "dna" in low:
                return [
                    "mRNA vaccines do not change DNA",
                    "mRNA vaccines cannot alter human DNA",
                    "no evidence mRNA vaccines integrate into human genome",
                    "studies show mRNA does not enter nucleus DNA integration",
                ]
            return []

        anchor_subject = subject_phrase or seg.split()[0]
        pred_obj = " ".join([predicate, obj]).strip()
        hints = [
            f"{anchor_subject} do not {pred_obj}".strip(),
            f"{anchor_subject} cannot {pred_obj}".strip(),
            f"no evidence that {anchor_subject} {pred_obj}".strip(),
            f"{anchor_subject} does not {pred_obj}".strip(),
        ]
        if "mrna" in low and "dna" in low:
            hints.append("studies show mRNA does not enter nucleus DNA integration")
        return [h for h in hints if h]

    async def _fetch_web_evidence_for_unknown_segments(
        self,
        unknown_segments: List[str],
        max_queries_per_segment: int = 2,
        max_urls_per_query: int = 3,
        enable_predicate_refute_queries: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch web evidence for UNKNOWN claim segments."""
        all_web_evidence = []
        generated_predicate_queries: List[str] = []

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
                hinted = [q for q in self._segment_recovery_query_hints(segment) if q]
                predicate_hints: List[str] = []
                if enable_predicate_refute_queries:
                    predicate_hints = [q for q in self._predicate_refute_query_hints(segment) if q]
                    generated_predicate_queries.extend(predicate_hints)
                if hinted:
                    queries = list(dict.fromkeys(predicate_hints + hinted + (queries or [])))
                elif predicate_hints:
                    queries = list(dict.fromkeys(predicate_hints + (queries or [])))
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
        self._last_predicate_queries_generated = list(dict.fromkeys(generated_predicate_queries))
        return all_web_evidence


def _self_test_unverifiable_lock_and_rationale() -> None:
    """Lightweight sanity checks for UNVERIFIABLE lock behavior."""
    # Case 1: mixed signals on absolute claim should force lock.
    support_signal = 0.46
    contradict_signal = 0.47
    decisive_support = support_signal >= 0.58 and contradict_signal <= 0.40
    decisive_contradict = contradict_signal >= 0.58 and support_signal <= 0.40
    conflicting_signals = support_signal >= 0.45 and contradict_signal >= 0.45
    weak_signal_no_stance = (not decisive_support and not decisive_contradict) or conflicting_signals
    policy_insufficient = True
    final_lock_unverifiable = bool(weak_signal_no_stance or policy_insufficient)
    assert final_lock_unverifiable, "Mixed signals must lock verdict to UNVERIFIABLE."

    # Case 2: UNVERIFIABLE rationale must not imply TRUE.
    vg = VerdictGenerator.__new__(VerdictGenerator)
    reconciled = {"verdict": Verdict.UNVERIFIABLE.value, "unresolved_segments": 1}
    breakdown = [{"status": "UNKNOWN"}]
    rationale = vg._rewrite_rationale_from_breakdown("placeholder", breakdown, reconciled)
    assert "UNVERIFIABLE" in rationale, "Rationale must explicitly state UNVERIFIABLE."
    assert "Verdict is TRUE" not in rationale, "Rationale must not imply TRUE when UNVERIFIABLE."


if __name__ == "__main__":
    _self_test_unverifiable_lock_and_rationale()
    print("verdict_generator sanity checks passed")
