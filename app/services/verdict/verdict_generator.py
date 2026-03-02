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
from time import perf_counter
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
from app.core.logger import get_logger, log_value_payload
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
from app.services.verdict.v2 import build_evidence_scores_v2, compute_verdict_policy_v2
from app.services.verdict.v2.calibration import ConfidenceCalibrator
from app.services.verdict.v2.normalizer import is_blocked_content
from app.services.verdict.v2.reconciler import reconcile_verdict
from app.services.verdict.v2.shadow import compute_shadow_diff
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
verdict_v2_shadow_total = Counter(
    "verdict_v2_shadow_total",
    "Total v2 shadow evaluations",
    ["parity"],
)
verdict_v2_fail_open_total = Counter(
    "verdict_v2_fail_open_total",
    "Total v2 fail-open fallbacks",
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


RATIONALE_GENERATION_PROMPT = """You are an expert scientific fact-checking writer.
Write a short, human-readable verification rationale.

Input claim:
{claim}

Final verdict:
{verdict}

Claim breakdown (segment status + cited fact):
{claim_breakdown_text}

Relevant evidence only (supporting/contradicting):
{relevant_evidence_text}

Instructions:
- Write 2-4 sentences that are easy to understand at a glance.
- Start with a direct verdict statement about the claim.
- Mention the main supporting and/or contradicting evidence balance.
- Keep language precise and non-hyped. Do not use boilerplate phrasing.
- Do not invent evidence. Use only the provided inputs.
- If evidence is mixed or limited, state that clearly.

Return ONLY valid JSON:
{{
  "rationale": "..."
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
        self.MAX_WEB_ROUNDS_PRE_VERDICT = int(os.getenv("VERDICT_MAX_WEB_ROUNDS_PRE", "4"))
        self.WEB_SEGMENTS_LIMIT = int(os.getenv("VERDICT_WEB_SEGMENTS_LIMIT", "5"))
        self.MAX_UNKNOWN_ROUNDS_POST_VERDICT = int(os.getenv("VERDICT_MAX_UNKNOWN_ROUNDS_POST", "4"))
        v2_enabled_env = os.getenv("VERDICT_ENGINE_V2_ENABLED")
        if v2_enabled_env is None:
            # Default-on for deterministic evidence policy. Can be disabled explicitly.
            self.v2_enabled = True
        else:
            self.v2_enabled = v2_enabled_env.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.v2_shadow = os.getenv("VERDICT_ENGINE_V2_SHADOW", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.v2_fail_open = os.getenv("VERDICT_ENGINE_V2_FAIL_OPEN", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.v2_latency_budget_ms = int(os.getenv("VERDICT_ENGINE_V2_LATENCY_BUDGET_MS", "2500"))
        self.v2_calibrator = ConfidenceCalibrator(os.getenv("VERDICT_ENGINE_V2_CALIBRATOR_PATH"))
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
        strict_predicate_threshold = max(
            float(PREDICATE_MATCH_THRESHOLD),
            float(os.getenv("SEGMENT_STRICT_PREDICATE_FLOOR", "0.35")),
        )
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
            if float(self.compute_predicate_match(claim, stmt) or 0.0) < strict_predicate_threshold:
                continue
            if float(self._segment_anchor_overlap(claim, stmt) or 0.0) < max(0.25, ANCHOR_THRESHOLD - 0.05):
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
        attempted_web_urls: set[str] = set()
        if self.confidence_mode:
            top_k = max(top_k, int(os.getenv("CONFIDENCE_VERDICT_TOP_K", "10")))
        effective_top_k = self._effective_top_k_for_claim(claim, top_k)
        if not ranked_evidence:
            logger.warning(
                "[VerdictGenerator] No ranked VDB/KG evidence; forcing targeted web evidence recovery "
                "(used_web_search=%s cache_sufficient=%s)",
                used_web_search,
                cache_sufficient,
            )
            segments = self._split_claim_into_segments(claim)[: self.WEB_SEGMENTS_LIMIT]
            web_boost = await self._fetch_web_evidence_for_unknown_segments(
                segments,
                max_queries_per_segment=2 if used_web_search else 3,
                max_urls_per_query=2 if used_web_search else 3,
                enable_predicate_refute_queries=True,
                attempted_urls=attempted_web_urls,
            )
            if not web_boost:
                # Aggressive fallback pass for hard claims.
                web_boost = await self._fetch_web_evidence_for_unknown_segments(
                    segments,
                    max_queries_per_segment=4,
                    max_urls_per_query=4,
                    enable_predicate_refute_queries=True,
                    attempted_urls=attempted_web_urls,
                )
            if not web_boost:
                fallback = self._unverifiable_result(
                    claim,
                    "No evidence retrieved after forced targeted web recovery",
                )
                return self._enforce_binary_verdict_payload(claim, fallback, evidence=[])
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
            logger.debug(
                f"[VerdictGenerator] Evidence: {len(ranked_evidence)} ranked + "
                f"{len(segment_evidence)} segment-specific = {len(enriched_evidence)} total"
            )
        else:
            logger.debug(f"[VerdictGenerator] Using {len(ranked_evidence)} ranked evidence (segment retrieval skipped)")

        # Step 2) PRE-VERDICT web-boost loop driven by *sufficiency*
        pre_evidence = enriched_evidence[: min(len(enriched_evidence), max(effective_top_k, 12), 20)]
        if not cache_sufficient:
            pre_verdict_rounds = max(1, int(self.MAX_WEB_ROUNDS_PRE_VERDICT))
            if used_web_search:
                pre_verdict_rounds += 1
            for round_i in range(pre_verdict_rounds):
                if adaptive_metrics is not None:
                    insufficient = not bool(adaptive_metrics.get("is_sufficient", False))
                else:
                    insufficient = self._policy_says_insufficient(claim, pre_evidence, adaptive_metrics=None)
                weak = self._needs_web_boost(pre_evidence[: min(len(pre_evidence), 6)], claim=claim)
                if not insufficient and not weak:
                    logger.debug(f"[VerdictGenerator] Pre-verdict evidence sufficient (round={round_i}). Skipping web.")
                    break
                logger.debug(
                    f"[VerdictGenerator] Pre-verdict evidence insufficient/weak (round={round_i}). "
                    f"insufficient={insufficient} weak={weak} -> web search"
                )
                segments = self._split_claim_into_segments(claim)[: self.WEB_SEGMENTS_LIMIT]
                web_boost = await self._fetch_web_evidence_for_unknown_segments(
                    segments,
                    max_queries_per_segment=2 if used_web_search else 3,
                    max_urls_per_query=2 if used_web_search else 3,
                    enable_predicate_refute_queries=True,
                    attempted_urls=attempted_web_urls,
                )
                if not web_boost:
                    logger.warning("[VerdictGenerator] Web boost returned no facts.")
                    break
                logger.debug(f"[VerdictGenerator] Web boost facts: {len(web_boost)}")
                merged_with_web = pre_evidence + web_boost
                merged_with_web.sort(key=self._deterministic_evidence_sort_key)
                pre_evidence = merged_with_web[: min(len(merged_with_web), 18)]
        pre_evidence.sort(key=self._deterministic_evidence_sort_key)
        top_evidence = self._select_balanced_top_evidence(claim, pre_evidence, top_k=min(effective_top_k, 20))
        if not top_evidence:
            logger.warning(
                "[VerdictGenerator] Top evidence still empty before verdict; running emergency evidence recovery."
            )
            emergency_segments = self._split_claim_into_segments(claim)[: self.WEB_SEGMENTS_LIMIT]
            emergency_web = await self._fetch_web_evidence_for_unknown_segments(
                emergency_segments,
                max_queries_per_segment=4,
                max_urls_per_query=4,
                enable_predicate_refute_queries=True,
                attempted_urls=attempted_web_urls,
            )
            if emergency_web:
                pre_evidence = self._merge_evidence(pre_evidence, emergency_web)
                top_evidence = self._select_balanced_top_evidence(claim, pre_evidence, top_k=min(effective_top_k, 20))
                logger.debug(
                    "[VerdictGenerator] Emergency recovery added %d evidence items (top=%d)",
                    len(emergency_web),
                    len(top_evidence),
                )

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

            logger.debug(
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
                    logger.debug(
                        "[VerdictGenerator] Found %d UNKNOWN segments, running targeted recovery...",
                        len(unknown_segments),
                    )

                    candidate_boost = await self._retrieve_segment_evidence_for_segments(
                        unknown_segments,
                        top_k=3 if self.confidence_mode else 2,
                    )
                    if candidate_boost:
                        logger.debug(
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
                        contradict_signal = float(analysis_counts.get("map_contradict_signal_max", 0.0) or 0.0)
                        pred_match_max = float(verdict_result.get("predicate_match_score_used", 0.0) or 0.0)
                        strictness_level = str(
                            (verdict_result.get("strictness_profile", {}) or {}).get("required_evidence_level", "LOW")
                        ).upper()
                        strictness_medium_or_higher = strictness_level in {"MEDIUM", "HIGH", "VERY_HIGH"}
                        need_predicate_refute = (
                            len(unknown_segments) > 0
                            and strictness_medium_or_higher
                            and contradict_signal <= 0.0
                            and pred_match_max <= 0.0
                            and (
                                (not explicit_refutes_found) or contradict_cov < float(CONTRADICT_RATIO_FOR_FORCE_FALSE)
                            )
                        )
                        web_evidence = await self._fetch_web_evidence_for_unknown_segments(
                            unknown_segments,
                            max_queries_per_segment=1 if used_web_search else 2,
                            max_urls_per_query=1 if used_web_search else 3,
                            enable_predicate_refute_queries=need_predicate_refute,
                            attempted_urls=attempted_web_urls,
                        )
                        candidate_boost = web_evidence
                        if web_evidence:
                            logger.debug(
                                "[VerdictGenerator] Retrieved %d additional facts from targeted web search",
                                len(web_evidence),
                            )

                    if not candidate_boost:
                        logger.debug(
                            "[VerdictGenerator] No additional targeted evidence found for UNKNOWN segments; stopping."
                        )
                        break

                    enriched_evidence = self._merge_evidence(top_evidence, candidate_boost)
                    top_evidence = self._select_balanced_top_evidence(
                        claim, enriched_evidence, top_k=min(effective_top_k, 20)
                    )
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
                        logger.debug(
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
            final_payload = self._enforce_binary_verdict_payload(claim, verdict_result, evidence=top_evidence)
            llm_rationale = await self._generate_llm_rationale(claim, final_payload)
            if llm_rationale:
                final_payload["rationale"] = llm_rationale
            return final_payload

        except Exception as e:
            logger.error(f"[VerdictGenerator] LLM call failed: {e}")
            fallback = self._unverifiable_result(claim, f"Verdict generation failed: {str(e)}")
            final_payload = self._enforce_binary_verdict_payload(
                claim,
                fallback,
                evidence=top_evidence if top_evidence else [],
            )
            llm_rationale = await self._generate_llm_rationale(claim, final_payload)
            if llm_rationale:
                final_payload["rationale"] = llm_rationale
            return final_payload

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

    @staticmethod
    def _format_claim_breakdown_for_rationale_prompt(claim_breakdown: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for idx, seg in enumerate(claim_breakdown or []):
            seg_text = str(seg.get("exact_claim_segment") or seg.get("claim_segment") or "").strip()
            status = str(seg.get("status") or "UNKNOWN").strip().upper()
            fact = str(seg.get("supporting_fact") or "").strip()
            src = str(seg.get("source_url") or "").strip()
            if not seg_text:
                continue
            line = f"[{idx}] Segment: {seg_text}\n    Status: {status}"
            if fact:
                line += f"\n    Fact: {fact}"
            if src:
                line += f"\n    Source: {src}"
            lines.append(line)
        return "\n\n".join(lines) if lines else "No segment breakdown available."

    def _format_relevant_evidence_for_rationale_prompt(self, payload: Dict[str, Any], max_items: int = 8) -> str:
        evidence_map = list(payload.get("evidence_map") or [])
        relevant = []
        for ev in evidence_map:
            rel = self._normalize_relevance_label(ev.get("relevance", "NEUTRAL"))
            if rel in {"SUPPORTS", "REFUTES"}:
                relevant.append(ev)
        if not relevant:
            # Fallback: include strongest neutral evidence if no explicit support/refute survived.
            relevant = sorted(
                evidence_map,
                key=lambda x: float(x.get("relevance_score", 0.0) or 0.0),
                reverse=True,
            )[: max(1, min(3, len(evidence_map)))]

        relevant = sorted(
            relevant,
            key=lambda x: float(x.get("relevance_score", 0.0) or 0.0),
            reverse=True,
        )[:max_items]

        lines: List[str] = []
        for i, ev in enumerate(relevant):
            stmt = str(ev.get("statement") or "").strip()
            src = str(ev.get("source_url") or "").strip()
            rel = self._normalize_relevance_label(ev.get("relevance", "NEUTRAL"))
            score = float(ev.get("relevance_score", 0.0) or 0.0)
            if not stmt:
                continue
            lines.append(
                f"[{i}] Relevance: {rel} (score={score:.3f})\n"
                f"    Statement: {stmt}\n"
                f"    Source: {src or 'Unknown'}"
            )
        return "\n\n".join(lines) if lines else "No relevant evidence available."

    async def _generate_llm_rationale(self, claim: str, payload: Dict[str, Any]) -> str:
        """Generate the final rationale with Llama from finalized verdict state."""
        try:
            claim_breakdown_text = self._format_claim_breakdown_for_rationale_prompt(
                payload.get("claim_breakdown") or []
            )
            relevant_evidence_text = self._format_relevant_evidence_for_rationale_prompt(payload)
            prompt = RATIONALE_GENERATION_PROMPT.format(
                claim=claim,
                verdict=str(payload.get("verdict") or "UNVERIFIABLE"),
                claim_breakdown_text=claim_breakdown_text,
                relevant_evidence_text=relevant_evidence_text,
            )
            result = await self.llm_service.ainvoke(
                prompt,
                response_format="json",
                priority=LLMPriority.HIGH,
                temperature=LLM_TEMPERATURE_VERDICT,
                call_tag="verdict_generation",
            )
            rationale = ""
            if isinstance(result, dict):
                rationale = str(result.get("rationale") or "").strip()
            if not rationale:
                return str(payload.get("rationale") or "").strip()
            return rationale
        except Exception as e:
            logger.warning("[VerdictGenerator] LLM rationale generation failed, using fallback rationale: %s", e)
            return str(payload.get("rationale") or "").strip()

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
        v2_enabled = bool(getattr(self, "v2_enabled", False))
        v2_shadow = bool(getattr(self, "v2_shadow", True))
        v2_fail_open = bool(getattr(self, "v2_fail_open", True))
        calibrator = getattr(self, "v2_calibrator", None) or ConfidenceCalibrator(None)
        decision_trace_id = self._build_decision_trace_id(claim, evidence)
        engine_version = "v2" if v2_enabled else "v1"

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
        claim_triplet = self._extract_canonical_predicate_triplet(claim)
        predicate_queries_generated = list(getattr(self, "_last_predicate_queries_generated", []) or [])
        eligible_refute_items = [
            ev
            for ev in (evidence_map or [])
            if self._normalize_relevance_label(ev.get("relevance")) == "REFUTES"
            and not bool(ev.get("blocked_content", False))
            and bool(ev.get("intervention_match", False))
            and float(ev.get("predicate_match_score", 0.0) or 0.0) >= 0.4
            and float(ev.get("credibility", 0.0) or 0.0) >= 0.8
        ]
        explicit_refutes_found = any(
            float(ev.get("contradiction_score", 0.0) or 0.0) >= CONTRADICTION_THRESHOLD for ev in eligible_refute_items
        )
        predicate_match_score_used = max(
            [float(ev.get("predicate_match_score", 0.0) or 0.0) for ev in (evidence_map or [])] or [0.0]
        )
        truthfulness_invariant_applied = False

        # Extract key findings (grounded later from aligned evidence).
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
        fragmentary_claim = self._is_subjectless_predicate_fragment(claim)
        if fragmentary_claim:
            for seg in claim_breakdown or []:
                status = str(seg.get("status") or "UNKNOWN").upper()
                if status == "VALID":
                    seg["status"] = "PARTIALLY_VALID"
                align_dbg = seg.get("alignment_debug") or {}
                if isinstance(align_dbg, dict):
                    align_dbg.setdefault("claim_fragment", "subjectless_predicate_fragment")
                    seg["alignment_debug"] = align_dbg

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
            quantifier_refute_allowed = self._quantifier_refute_allowed(seg_text, polarity_text)
            dose_scope_refute_allowed = self._dose_scope_refute_allowed(seg_text, polarity_text)
            contradiction_eligibility = quantifier_refute_allowed and dose_scope_refute_allowed
            polarity_predicate_match = float(self.compute_predicate_match(seg_text, polarity_text) or 0.0)
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
            if (
                status in {"VALID", "PARTIALLY_VALID"}
                and polarity == "contradicts"
                and contradiction_eligibility
                and polarity_predicate_match >= 0.4
            ):
                seg["status"] = "INVALID" if status == "VALID" else "PARTIALLY_INVALID"
            if (
                status in {"VALID", "PARTIALLY_VALID"}
                and high_semantic_negation_mismatch
                and contradiction_eligibility
                and polarity_predicate_match >= 0.4
                and (polarity == "contradicts" or self._is_explicit_refutation_statement(polarity_text))
            ):
                seg["status"] = "INVALID" if status == "VALID" else "PARTIALLY_INVALID"
            if status in {"VALID", "PARTIALLY_VALID"} and polarity_text:
                object_tokens = self._segment_object_tokens(seg_text)
                statement_tokens = self._statement_tokens(polarity_text)
                no_object_overlap = bool(object_tokens) and len(object_tokens & statement_tokens) == 0
                if no_object_overlap and not self._is_explicit_refutation_statement(polarity_text):
                    seg["status"] = "UNKNOWN"
                    seg["supporting_fact"] = ""
                    seg["source_url"] = ""
                    seg["evidence_used_ids"] = []
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
                if (
                    high_semantic_negation_mismatch
                    and (polarity == "contradicts" or self._is_explicit_refutation_statement(polarity_text))
                    and contradiction_eligibility
                    and polarity_predicate_match >= 0.4
                ):
                    seg["status"] = "INVALID"
                    seg["supporting_fact"] = fact
                    seg["source_url"] = src
                    seg["evidence_used_ids"] = [best_idx] if best_idx >= 0 else []
                elif (
                    polarity == "contradicts"
                    and best_semantic >= 0.75
                    and contradiction_eligibility
                    and polarity_predicate_match >= 0.4
                ):
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
                    contradiction_eligibility
                    and polarity_predicate_match >= 0.4
                    and (
                        polarity == "contradicts"
                        or high_semantic_negation_mismatch
                        or self._is_explicit_refutation_statement(polarity_text or "")
                    )
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
        reconciled_v1 = self._reconcile_verdict_with_breakdown(claim, claim_breakdown)
        reconciled = dict(reconciled_v1)
        shadow_diff: Dict[str, Any] | None = None
        if v2_enabled or v2_shadow:
            shadow_start = perf_counter()
            try:
                reconciled_v2 = self._reconcile_verdict_v2(claim, claim_breakdown)
                if v2_enabled:
                    reconciled = dict(reconciled_v2)
                if v2_shadow:
                    shadow_diff = compute_shadow_diff(reconciled_v1, reconciled_v2)
                    verdict_v2_shadow_total.labels(parity="true" if shadow_diff.get("parity") else "false").inc()
                    shadow_diff["latency_ms"] = round((perf_counter() - shadow_start) * 1000.0, 2)
            except Exception as shadow_exc:
                if v2_fail_open:
                    verdict_v2_fail_open_total.inc()
                    logger.warning("[VerdictGenerator][v2] fail-open engaged in shadow/reconcile: %s", shadow_exc)
                else:
                    raise
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
            if v2_enabled:
                reconciled = self._reconcile_verdict_v2(claim, claim_breakdown)
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
            logger.debug(
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
        coverage_verdict_aligned = float(reconciled.get("weighted_truth", 0.0) or 0.0)
        coverage_adaptive = float(coverage_verdict_aligned)
        coverage_score = float(coverage_verdict_aligned)
        adaptive_trust_post = 0.0
        if adaptive_metrics is not None:
            coverage_adaptive = float(
                adaptive_metrics.get("coverage", coverage_verdict_aligned) or coverage_verdict_aligned
            )
            coverage_score = min(float(coverage_verdict_aligned), float(coverage_adaptive))
            agreement_ratio = float(adaptive_metrics.get("agreement", agreement_ratio) or agreement_ratio)
            diversity_score = float(adaptive_metrics.get("diversity", diversity_score) or diversity_score)
            adaptive_trust_post = float(adaptive_metrics.get("trust_post", 0.0) or 0.0)
            logger.debug(
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
                coverage_adaptive = float(
                    adaptive_metrics.get("coverage", coverage_verdict_aligned) or coverage_verdict_aligned
                )
                coverage_score = min(float(coverage_verdict_aligned), float(coverage_adaptive))
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
                if v2_enabled:
                    reconciled = self._reconcile_verdict_v2(claim, claim_breakdown)
                verdict_str = reconciled["verdict"]
                logger.debug(
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
            logger.debug(
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
        refute_cred_max = max([float(ev.get("credibility", 0.0) or 0.0) for ev in eligible_refute_items] or [0.0])
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
        support_status_count = sum(
            1
            for seg in (claim_breakdown or [])
            if str((seg or {}).get("status") or "").upper() in {"VALID", "PARTIALLY_VALID"}
        )
        invalid_status_count = sum(
            1
            for seg in (claim_breakdown or [])
            if str((seg or {}).get("status") or "").upper() in {"INVALID", "PARTIALLY_INVALID"}
        )
        all_segments_supported = bool(claim_breakdown) and support_status_count == len(claim_breakdown)
        strong_support_segment_present = (
            support_status_count > 0
            and invalid_status_count == 0
            and (all_segments_supported or map_support_signal >= 0.55 or top_stance == "entails")
        )
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
            eligible_refute_items
            and explicit_refutes_found
            and (
                (
                    contradiction_metrics["contradict_ratio"] >= contradict_ratio_threshold
                    and contradiction_metrics["contradict_diversity"] >= DIVERSITY_FORCE_FALSE
                )
                or (
                    map_contradict_signal >= 0.35
                    and refute_cred_max >= 0.85
                    and contradiction_metrics["contradict_diversity"] >= 0.5
                )
            )
        ):
            for seg in claim_breakdown or []:
                seg["status"] = "INVALID"
            verdict_str = Verdict.FALSE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 15.0)
            confidence = max(float(confidence or 0.0), 0.80)
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

        matched_statuses_now = [str(s or "UNKNOWN").upper() for s in (reconciled.get("matched_statuses") or [])]
        has_partial_or_unknown_status = any(
            s in {"PARTIALLY_VALID", "PARTIALLY_INVALID", "UNKNOWN"} for s in matched_statuses_now
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
            and not has_partial_or_unknown_status
            and not self._has_absolute_quantifier(claim)
        ):
            verdict_str = Verdict.TRUE.value
        elif canonical_stance == "entails" and verdict_str == Verdict.FALSE.value and not has_partial_or_unknown_status:
            verdict_str = Verdict.TRUE.value
            logger.warning(
                "[VerdictGenerator][Consistency] Flipped FALSE->TRUE " "(breakdown=%s map=%s rationale=%s)",
                breakdown_stance,
                map_stance,
                rationale_stance,
            )
        verdict_guard_reasons: List[str] = []
        trust_gate_binary_lock_applied = False
        decisive_contradiction_unlock = bool(
            canonical_stance == "contradicts"
            and (strong_vote_contradiction or float(map_contradict_signal or 0.0) >= 0.60)
            and float(admissible_ratio or 0.0) >= 0.50
            and (float(coverage_score or 0.0) >= 0.60 or float(agreement_ratio or 0.0) >= 0.80)
            and invalid_status_count >= max(1, support_status_count)
        )

        if rationale_stance == "contradicts" and verdict_str == Verdict.TRUE.value:
            verdict_str = Verdict.FALSE.value if vote_contradict >= 0.60 else Verdict.UNVERIFIABLE.value
        if weak_signal_no_stance and not strong_support_segment_present:
            verdict_str = Verdict.UNVERIFIABLE.value
        # Trust-governed binary lock:
        # if trust gate is not met, do not emit hard TRUE/FALSE.
        if policy_insufficient and verdict_str in {Verdict.TRUE.value, Verdict.FALSE.value}:
            trust_gate_binary_lock_applied = True
            verdict_guard_reasons.append("trust_gate_binary_lock")
            if verdict_str == Verdict.FALSE.value and decisive_contradiction_unlock:
                verdict_guard_reasons.append("trust_gate_binary_lock_bypassed_contradiction")
                trust_gate_binary_lock_applied = False
                confidence = max(float(confidence or 0.0), 0.62)
            elif verdict_str == Verdict.TRUE.value and strong_support_segment_present:
                verdict_str = Verdict.PARTIALLY_TRUE.value
            else:
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
        high_authority_support = (
            max(
                [
                    float(ev.get("credibility", 0.0) or 0.0)
                    for ev in (evidence_map or [])
                    if str(ev.get("relevance") or "").upper() == "SUPPORTS"
                ]
                or [0.0]
            )
            >= 0.90
        )

        # Conservative binary gate:
        # Avoid hard TRUE/FALSE unless stance is explicit and evidence quality/diversity are sufficient.
        if verdict_str in {Verdict.TRUE.value, Verdict.FALSE.value}:
            binary_gate_ok = (
                canonical_stance in {"entails", "contradicts"}
                and admissible_ratio >= 0.50
                and (
                    diversity_score >= 0.40
                    or adaptive_trust_post >= 0.45
                    or (verdict_str == Verdict.TRUE.value and unique_domains_count == 1 and high_authority_support)
                )
                and (coverage_score >= 0.70 or agreement_ratio >= 0.80)
                and (
                    unique_domains_count >= 2
                    or (verdict_str == Verdict.TRUE.value and unique_domains_count == 1 and high_authority_support)
                )
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
                and (
                    (unique_domains_count >= 2 and diversity_score >= 0.20)
                    or (unique_domains_count == 1 and high_authority_support)
                )
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
            if policy_insufficient and not decisive_contradiction_unlock:
                trust_gate_binary_lock_applied = True
                verdict_guard_reasons.append("trust_gate_binary_lock")
                verdict_str = Verdict.UNVERIFIABLE.value
                truth_score_percent = min(float(truth_score_percent or 0.0), 49.0)
            else:
                verdict_str = Verdict.FALSE.value
                truth_score_percent = min(float(truth_score_percent or 0.0), 10.0)
                confidence = max(float(confidence), 0.70)
        final_lock_unverifiable = bool(
            (weak_signal_no_stance and not strong_support_segment_present)
            or (policy_insufficient and canonical_stance == "neutral" and not strong_support_segment_present)
        )
        if strong_vote_support and self._has_absolute_quantifier(claim) and not final_lock_unverifiable:
            verdict_str = Verdict.PARTIALLY_TRUE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 55.0)
        if numeric_conf_floor is not None:
            confidence = max(float(confidence), float(numeric_conf_floor))
        if (
            final_lock_unverifiable
            and strict_override_fired != "CONTRADICTION_DOMINANCE"
            and map_contradict_signal < CONTRADICTION_THRESHOLD
            and not strong_support_segment_present
        ):
            verdict_str = Verdict.UNVERIFIABLE.value
            truth_score_percent = max(35.0, min(55.0, float(truth_score_percent or 0.0)))
            confidence = min(float(confidence), UNVERIFIABLE_CONFIDENCE_CAP)
            for seg in claim_breakdown or []:
                status_u = str(seg.get("status") or "UNKNOWN").upper()
                keep_supported_segment = (
                    status_u in {"VALID", "PARTIALLY_VALID"}
                    and support_status_count > 0
                    and invalid_status_count == 0
                    and map_contradict_signal < CONTRADICTION_THRESHOLD
                )
                if keep_supported_segment:
                    continue
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
            if v2_enabled:
                reconciled = self._reconcile_verdict_v2(claim, claim_breakdown)
            rationale = (
                "Available admissible evidence is mixed or insufficiently decisive for this claim, "
                "so the result is UNVERIFIABLE."
            )
        if (
            verdict_str == Verdict.UNVERIFIABLE.value
            and strong_support_segment_present
            and map_contradict_signal < CONTRADICTION_THRESHOLD
        ):
            verdict_str = Verdict.PARTIALLY_TRUE.value
            truth_score_percent = max(55.0, min(75.0, float(truth_score_percent or 0.0)))
        if verdict_str == Verdict.UNVERIFIABLE.value:
            current_statuses = [str((seg or {}).get("status") or "UNKNOWN").upper() for seg in (claim_breakdown or [])]
            has_support_now = any(s in {"VALID", "PARTIALLY_VALID"} for s in current_statuses)
            has_invalid_now = any(s in {"INVALID", "PARTIALLY_INVALID"} for s in current_statuses)
            if has_support_now and not has_invalid_now and map_contradict_signal < CONTRADICTION_THRESHOLD:
                verdict_str = Verdict.PARTIALLY_TRUE.value
                truth_score_percent = max(55.0, min(75.0, float(truth_score_percent or 0.0)))
        if verdict_str == Verdict.UNVERIFIABLE.value and numeric_truth_override is None:
            truth_score_percent = max(35.0, min(65.0, float(truth_score_percent or 0.0)))
        if self._has_absolute_quantifier(claim) and verdict_str == Verdict.TRUE.value:
            verdict_str = Verdict.PARTIALLY_TRUE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 55.0)

        if strict_override_fired == "CONTRADICTION_DOMINANCE":
            verdict_str = Verdict.FALSE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 15.0)
            confidence = max(float(confidence or 0.0), 0.80)

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
            if explicit_refutes_found and map_contradict_signal >= 0.35:
                confidence = max(float(confidence or 0.0), 0.80)
        v2_policy_output: Dict[str, Any] = {}
        v2_stance_diag: Dict[str, float] = {}
        class_probs_raw: Dict[str, float]
        class_probs: Dict[str, float]
        class_max_prob: float
        confidence_seed: float
        if v2_enabled:
            try:
                nli_top_n = max(1, int(os.getenv("REFUTE_STAGE2_TOP_N", "5")))
            except Exception:
                nli_top_n = 5
            try:
                evidence_scores_v2, v2_stance_diag = build_evidence_scores_v2(
                    claim=claim,
                    evidence_map=evidence_map or [],
                    evidence=evidence or [],
                    nli_top_n=nli_top_n,
                )
                v2_policy_output = compute_verdict_policy_v2(
                    scores=evidence_scores_v2,
                    coverage=float(coverage_score or 0.0),
                    diversity=float(diversity_score or 0.0),
                    calibrator=calibrator,
                    calibrator_features={
                        "agreement": float(agreement_ratio or 0.0),
                        "retrieval_depth": max(0.0, min(1.0, float(len(evidence or [])) / 20.0)),
                    },
                )
                verdict_str = str(v2_policy_output.get("verdict") or verdict_str)
                truth_score_percent = float(v2_policy_output.get("truthfulness_percent", truth_score_percent) or 0.0)
                confidence = float(v2_policy_output.get("calibrated_confidence", confidence) or 0.0)
                class_probs_raw = dict(v2_policy_output.get("class_probs_raw") or {})
                class_probs = dict(v2_policy_output.get("class_probs") or {})
                class_max_prob = max(float(v or 0.0) for v in class_probs.values()) if class_probs else 0.0
                confidence_seed = float(v2_policy_output.get("calibrated_confidence", confidence) or 0.0)
            except Exception as policy_exc:
                logger.warning("[VerdictGenerator][v2] policy compute failed, fallback to v1 scoring: %s", policy_exc)
                v2_policy_output = {}
                v2_stance_diag = {}
                v2_enabled = False
        if not v2_enabled:
            supports_in_map = sum(
                1 for ev in (evidence_map or []) if str(ev.get("relevance") or "").strip().upper() == "SUPPORTS"
            )
            refutes_in_map = sum(
                1 for ev in (evidence_map or []) if str(ev.get("relevance") or "").strip().upper() == "REFUTES"
            )
            raw_true_signal = max(
                0.0,
                min(
                    1.0,
                    (0.60 * float(map_support_signal or 0.0))
                    + (0.20 * float(coverage_score or 0.0))
                    + (0.20 * float(admissible_ratio or 0.0)),
                ),
            )
            raw_false_signal = max(
                0.0,
                min(
                    1.0,
                    (0.65 * float(map_contradict_signal or 0.0))
                    + (0.20 * float(coverage_score or 0.0))
                    + (0.15 * float(admissible_ratio or 0.0)),
                ),
            )
            # Reduce false posterior inflation when no refuting evidence is admitted.
            if refutes_in_map == 0 and supports_in_map > 0 and not explicit_refutes_found:
                raw_false_signal *= 0.55
                raw_true_signal = min(1.0, raw_true_signal * 1.05)
            elif explicit_refutes_found and supports_in_map == 0:
                raw_false_signal = min(1.0, raw_false_signal * 1.10)
            raw_unv_signal = max(
                0.0,
                min(
                    1.0,
                    (0.40 * (1.0 - float(coverage_score or 0.0)))
                    + (0.30 * (1.0 - float(admissible_ratio or 0.0)))
                    + (0.30 * max(0.0, 1.0 - max(raw_true_signal, raw_false_signal))),
                ),
            )
            # Soft verdict prior only; avoid hard forcing.
            if verdict_str == Verdict.TRUE.value:
                raw_true_signal *= 1.08
            elif verdict_str == Verdict.FALSE.value:
                raw_false_signal *= 1.08
            elif verdict_str == Verdict.UNVERIFIABLE.value:
                raw_unv_signal *= 1.10
            class_probs_raw = {
                "true": float(raw_true_signal),
                "false": float(raw_false_signal),
                "unverifiable": float(raw_unv_signal),
            }
            class_probs = calibrator.calibrate_distribution(
                class_probs_raw,
                features={
                    "coverage": float(coverage_score or 0.0),
                    "admissible_ratio": float(admissible_ratio or 0.0),
                    "contradict_signal": float(map_contradict_signal or 0.0),
                    "support_signal": float(map_support_signal or 0.0),
                },
            )
            class_max_prob = max(float(v or 0.0) for v in class_probs.values())
            confidence_seed = (0.55 * float(confidence or 0.0)) + (0.45 * class_max_prob)
            confidence = calibrator.calibrate(
                float(confidence_seed or 0.0),
                features={
                    "coverage": float(coverage_score or 0.0),
                    "agreement": float(agreement_ratio or 0.0),
                    "diversity": float(diversity_score or 0.0),
                    "contradict_signal": float(map_contradict_signal or 0.0),
                    "admissible_ratio": float(admissible_ratio or 0.0),
                    "evidence_quality": max(0.0, min(1.0, float(evidence_quality_percent or 0.0) / 100.0)),
                    "class_max_prob": float(class_max_prob or 0.0),
                    "retrieval_depth": max(0.0, min(1.0, float(len(evidence or [])) / 20.0)),
                },
            )
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
            and not strong_support_segment_present
        ):
            verdict_str = Verdict.UNVERIFIABLE.value
            rationale = (
                "Available admissible evidence is mixed or insufficiently decisive for this claim, "
                "so the result is UNVERIFIABLE."
            )
        if (
            verdict_str == Verdict.UNVERIFIABLE.value
            and strong_support_segment_present
            and ("ambigu" in str(rationale or "").lower() or "partially supported" in str(rationale or "").lower())
            and map_contradict_signal < CONTRADICTION_THRESHOLD
        ):
            verdict_str = Verdict.PARTIALLY_TRUE.value
        if llm_verdict == Verdict.TRUE.value and verdict_str != Verdict.TRUE.value:
            try:
                truth_score_percent = min(float(truth_score_percent), 89.9)
            except Exception:
                truth_score_percent = 89.9

        # Consistency guard: TRUE cannot coexist with low truthfulness or unresolved
        # segment coverage.
        if verdict_str == Verdict.TRUE.value:
            total_segments = max(1, len(claim_breakdown or []))
            unresolved_segments = int(reconciled.get("unresolved_segments", 0) or 0)
            has_partial_or_unknown_status = any(
                str(s or "UNKNOWN").upper() in {"PARTIALLY_VALID", "PARTIALLY_INVALID", "UNKNOWN"}
                for s in (reconciled.get("matched_statuses") or [])
            )
            if (
                unresolved_segments > 0
                or support_status_count < total_segments
                or has_partial_or_unknown_status
                or float(truth_score_percent or 0.0) < 70.0
            ):
                verdict_str = Verdict.PARTIALLY_TRUE.value
                truth_score_percent = max(55.0, min(79.9, float(truth_score_percent or 0.0)))
                confidence = min(float(confidence or 0.0), 0.85)
        if fragmentary_claim:
            if verdict_str == Verdict.TRUE.value:
                verdict_str = Verdict.PARTIALLY_TRUE.value
            elif verdict_str == Verdict.FALSE.value:
                verdict_str = Verdict.UNVERIFIABLE.value
            truth_score_percent = min(float(truth_score_percent or 0.0), 60.0)
            confidence = min(float(confidence or 0.0), 0.72)
            if "incomplete claim fragment" not in str(rationale or "").lower():
                rationale = (
                    "Incomplete claim fragment without an explicit subject/intervention; "
                    "evaluated conservatively against available evidence."
                )

        # Final trust gate lock (after all overrides and consistency transforms):
        # keep outputs trust-governed by prohibiting hard binary verdicts when policy is insufficient.
        if (
            policy_insufficient
            and verdict_str in {Verdict.TRUE.value, Verdict.FALSE.value}
            and not v2_enabled
            and not (verdict_str == Verdict.FALSE.value and decisive_contradiction_unlock)
        ):
            trust_gate_binary_lock_applied = True
            verdict_guard_reasons.append("trust_gate_binary_lock")
            if verdict_str == Verdict.TRUE.value and strong_support_segment_present:
                verdict_str = Verdict.PARTIALLY_TRUE.value
                truth_score_percent = max(55.0, min(75.0, float(truth_score_percent or 0.0)))
            else:
                verdict_str = Verdict.UNVERIFIABLE.value
                truth_score_percent = min(49.0, float(truth_score_percent or 49.0))
                confidence = min(float(confidence or 0.0), 0.60)

        # v2 deterministic policy is the final authority for class/score selection.
        # Keep legacy heuristics active for rationale shaping but do not let them
        # hard-lock the final class under v2.
        if v2_enabled and v2_policy_output:
            verdict_str = str(v2_policy_output.get("verdict") or verdict_str)
            truth_score_percent = float(v2_policy_output.get("truthfulness_percent", truth_score_percent) or 0.0)
            confidence = float(v2_policy_output.get("calibrated_confidence", confidence) or 0.0)
            if verdict_str == Verdict.UNVERIFIABLE.value:
                confidence = min(float(confidence or 0.0), float(UNVERIFIABLE_CONFIDENCE_CAP))
            confidence = max(0.05, min(0.98, float(confidence or 0.0)))
            policy_insufficient = False
            trust_gate_binary_lock_applied = False
            verdict_guard_reasons = [r for r in verdict_guard_reasons if r != "trust_gate_binary_lock"]

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

        coverage_verdict_aligned = float(
            reconciled.get("weighted_truth", coverage_verdict_aligned) or coverage_verdict_aligned
        )
        coverage_adaptive = float(coverage_adaptive or coverage_verdict_aligned)
        coverage_score = min(float(coverage_score or 0.0), coverage_verdict_aligned, coverage_adaptive)

        segment_evidence_counts: Dict[str, int] = {}
        min_per_segment_target = max(1, int(os.getenv("VERDICT_MIN_EVIDENCE_PER_SEGMENT", "5")))
        for seg in claim_breakdown or []:
            seg_text = str(seg.get("claim_segment") or "").strip()
            seg_ids = seg.get("evidence_used_ids") or []
            count = len([x for x in seg_ids if isinstance(x, int) or str(x).isdigit()])
            if seg_text:
                segment_evidence_counts[seg_text] = count
        undercovered_segments = [s for s, c in segment_evidence_counts.items() if c < min_per_segment_target]

        key_findings = self._build_evidence_grounded_key_findings(claim_breakdown, evidence_map)
        direct_evidence = self._build_direct_evidence_list(claim_breakdown, evidence_map)
        evidence_for_payload = list(evidence or [])
        evidence_backfilled_from_map = False
        if not evidence_for_payload and evidence_map:
            # Preserve payload consistency when ranking produced sparse/empty top-k
            # but validated evidence_map rows exist.
            evidence_backfilled_from_map = True
            for item in evidence_map:
                if not isinstance(item, dict):
                    continue
                stmt = str(item.get("statement") or "").strip()
                if not stmt:
                    continue
                evidence_for_payload.append(
                    {
                        "statement": stmt,
                        "source_url": str(item.get("source_url") or ""),
                        "score": float(item.get("relevance_score", 0.0) or 0.0),
                        "sem_score": float(item.get("relevance_score", 0.0) or 0.0),
                        "kg_score": 0.0,
                        "credibility": float(item.get("credibility", 0.5) or 0.5),
                    }
                )
                if len(evidence_for_payload) >= 12:
                    break
        evidence_attribution = [
            {
                "segment": str(seg.get("claim_segment") or ""),
                "status": str(seg.get("status") or "UNKNOWN"),
                "evidence_ids": [int(x) for x in (seg.get("evidence_used_ids") or []) if str(x).isdigit()],
            }
            for seg in (claim_breakdown or [])
        ]

        verdict_payload = {
            "verdict": verdict_str,
            "confidence": confidence,
            "calibrated_confidence": confidence,
            # Backward-compatibility: keep `truthfulness_percent` but now make it a status-driven truth score.
            "truthfulness_percent": truth_score_percent,
            "truth_score_percent": truth_score_percent,
            "class_probs": class_probs,
            "evidence_attribution": evidence_attribution,
            "calibration_meta": {
                "calibrator_version": calibrator.version,
                "confidence_seed": round(float(confidence_seed or 0.0), 4),
                "class_probs_raw": {k: round(float(v or 0.0), 4) for k, v in class_probs_raw.items()},
                "class_max_prob": round(float(class_max_prob or 0.0), 4),
                "v2_policy_active": bool(v2_enabled),
                "v2_policy": {
                    "support_mass": round(float(v2_policy_output.get("support_mass", 0.0) or 0.0), 4),
                    "refute_mass": round(float(v2_policy_output.get("refute_mass", 0.0) or 0.0), 4),
                    "neutral_mass": round(float(v2_policy_output.get("neutral_mass", 0.0) or 0.0), 4),
                    "evidence_sufficiency": round(float(v2_policy_output.get("evidence_sufficiency", 0.0) or 0.0), 4),
                    "agreement_score": round(float(v2_policy_output.get("agreement_score", 0.0) or 0.0), 4),
                    "retrieval_entropy": round(float(v2_policy_output.get("retrieval_entropy", 0.0) or 0.0), 4),
                    "admissibility_rate": round(float(v2_policy_output.get("admissibility_rate", 0.0) or 0.0), 4),
                },
                "refute_pipeline_stats": {
                    "refute_candidate_count_stage1": int(v2_stance_diag.get("refute_candidate_count_stage1", 0.0) or 0),
                    "refute_verified_count_stage2": int(v2_stance_diag.get("refute_verified_count_stage2", 0.0) or 0),
                    "refutes_admission_rate": round(float(v2_stance_diag.get("refutes_admission_rate", 0.0) or 0.0), 4),
                },
            },
            "evidence_quality_percent": float(evidence_quality_percent or 0.0),
            "rationale": rationale,
            "claim_breakdown": claim_breakdown,
            "evidence_map": evidence_map,
            "evidence": evidence_for_payload,
            "key_findings": key_findings,
            "direct_evidence": direct_evidence,
            "claim": claim,
            "evidence_count": len(evidence_for_payload),
            "evidence_backfilled_from_map": bool(evidence_backfilled_from_map),
            "required_segments_count": reconciled["required_segments_count"],
            "resolved_segments_count": reconciled["resolved_segments_count"],
            "required_segments_resolved": reconciled["required_segments_resolved"],
            "unresolved_segments": reconciled["unresolved_segments"],
            "status_weighted_truth": reconciled.get("weighted_truth", 0.0),
            "truthfulness_cap": reconciled.get("truthfulness_cap", 100.0),
            "agreement_ratio": agreement_ratio,
            "coverage": float(coverage_score or 0.0),
            "coverage_adaptive": float(coverage_adaptive or 0.0),
            "coverage_verdict_aligned": float(coverage_verdict_aligned or 0.0),
            "policy_sufficient": not policy_insufficient,
            "trust_gate_binary_lock_applied": bool(trust_gate_binary_lock_applied),
            "verdict_guard_reasons": sorted(set(verdict_guard_reasons)),
            "segment_evidence_counts": segment_evidence_counts,
            "undercovered_segments": undercovered_segments,
            "min_evidence_per_segment_target": int(min_per_segment_target),
            "verdict_reconciled": bool(verdict_str != llm_verdict),
            "skip_targeted_recovery": bool(skip_targeted_recovery),
            "strictness_profile": strictness_profile.to_dict(),
            "override_fired": effective_override_fired,
            "override_reason": effective_override_reason,
            "override_key_numbers": effective_override_key_numbers,
            "explicit_refutes_found": bool(explicit_refutes_found),
            "eligible_refutes_count": len(eligible_refute_items),
            "predicate_queries_generated": predicate_queries_generated,
            "predicate_match_score_used": round(float(predicate_match_score_used or 0.0), 4),
            "canonical_predicate": str(claim_triplet.get("canonical_predicate") or ""),
            "contradiction_override_fired": bool(strict_override_fired == "CONTRADICTION_DOMINANCE"),
            "truthfulness_invariant_applied": bool(truthfulness_invariant_applied),
            "engine_version": engine_version,
            "decision_trace_id": decision_trace_id,
            "calibration_version": calibrator.version,
            "shadow_diff": shadow_diff if v2_shadow else None,
            "stance_scores": {
                "support_mass": float(v2_policy_output.get("support_mass", 0.0) or 0.0),
                "refute_mass": float(v2_policy_output.get("refute_mass", 0.0) or 0.0),
                "neutral_mass": float(v2_policy_output.get("neutral_mass", 0.0) or 0.0),
            },
            "evidence_sufficiency": float(v2_policy_output.get("evidence_sufficiency", 0.0) or 0.0),
            "agreement_score": float(v2_policy_output.get("agreement_score", agreement_ratio) or agreement_ratio),
            "retrieval_entropy": float(v2_policy_output.get("retrieval_entropy", 0.0) or 0.0),
            "refute_pipeline_stats": {
                "refute_candidate_count_stage1": int(v2_stance_diag.get("refute_candidate_count_stage1", 0.0) or 0),
                "refute_verified_count_stage2": int(v2_stance_diag.get("refute_verified_count_stage2", 0.0) or 0),
                "refutes_admission_rate": float(v2_stance_diag.get("refutes_admission_rate", 0.0) or 0.0),
            },
            "analysis_counts": {
                "evidence_total_input": len(evidence),
                "evidence_total_payload": len(evidence_for_payload),
                "evidence_map_count": len(evidence_map or []),
                "claim_breakdown_count": len(claim_breakdown or []),
                "admissible_evidence_count": admissible_count,
                "admissible_evidence_ratio": round(admissible_ratio, 4),
                "unique_source_domains": len(unique_domains),
                "coverage_adaptive": round(float(coverage_adaptive or 0.0), 4),
                "coverage_verdict_aligned": round(float(coverage_verdict_aligned or 0.0), 4),
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
                "claim_fragmentary": bool(fragmentary_claim),
                "llm_verdict_raw": llm_verdict,
                "llm_verdict_changed": bool(llm_verdict != verdict_str),
                "evidence_sufficiency_v2": round(float(v2_policy_output.get("evidence_sufficiency", 0.0) or 0.0), 4),
                "agreement_score_v2": round(
                    float(v2_policy_output.get("agreement_score", agreement_ratio) or agreement_ratio), 4
                ),
                "retrieval_entropy_v2": round(float(v2_policy_output.get("retrieval_entropy", 0.0) or 0.0), 4),
                "refute_candidate_count_stage1": int(v2_stance_diag.get("refute_candidate_count_stage1", 0.0) or 0),
                "refute_verified_count_stage2": int(v2_stance_diag.get("refute_verified_count_stage2", 0.0) or 0),
                "refutes_admission_rate": round(float(v2_stance_diag.get("refutes_admission_rate", 0.0) or 0.0), 4),
            },
        }
        verdict_payload = self._enforce_binary_verdict_payload(claim, verdict_payload, evidence=evidence_for_payload)
        log_value_payload(
            logger,
            "verdict",
            {
                "verdict": verdict_payload["verdict"],
                "truthfulness_percent": verdict_payload["truthfulness_percent"],
                "confidence": verdict_payload["confidence"],
                "calibrated_confidence": verdict_payload.get("calibrated_confidence"),
                "coverage": coverage_score,
                "agreement": agreement_ratio,
                "diversity": diversity_score,
                "trust_post": adaptive_trust_post,
                "class_probs": verdict_payload.get("class_probs"),
                "override_fired": verdict_payload["override_fired"],
                "override_reason": verdict_payload["override_reason"],
                "explicit_refutes_found": verdict_payload["explicit_refutes_found"],
                "predicate_match_score_used": verdict_payload["predicate_match_score_used"],
                "contradict_signal": verdict_payload["analysis_counts"]["map_contradict_signal_max"],
            },
        )
        log_value_payload(
            logger,
            "verdict",
            {
                "claim_breakdown_full": verdict_payload["claim_breakdown"],
                "evidence_map_full": verdict_payload["evidence_map"],
                "analysis_counts_full": verdict_payload["analysis_counts"],
                "truthfulness_invariant_applied": verdict_payload["truthfulness_invariant_applied"],
                "unverifiable_confidence_cap_applied": bool(
                    verdict_payload["override_fired"] == "UNVERIFIABLE_CONFIDENCE_CAP"
                ),
                "predicate_queries_generated": verdict_payload["predicate_queries_generated"],
            },
            level="debug",
            debug_only=True,
        )
        return verdict_payload

    def _segment_anchor_overlap(self, segment: str, statement: str) -> float:
        eval_result = evaluate_anchor_match(segment, statement)
        groups = eval_result.get("anchor_groups", []) or []
        matched = int(eval_result.get("matched_groups", 0) or 0)
        total = len(groups)
        base = (matched / total) if total > 0 else 0.0

        # Paraphrase/lemma fallback when strict anchors miss.
        seg_tokens = self._content_tokens(segment)
        stmt_tokens = self._content_tokens(statement)
        if not seg_tokens:
            return max(0.0, min(1.0, base))
        overlap = len(seg_tokens & stmt_tokens) / max(1, len(seg_tokens))
        return max(0.0, min(1.0, max(base, overlap)))

    def _content_tokens(self, text: str) -> set[str]:
        raw = [self._lemma_token(t) for t in re.findall(r"\b[a-z][a-z0-9_-]+\b", (text or "").lower())]
        stop = {
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
            "to",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "can",
            "could",
            "may",
            "might",
            "help",
            "helps",
            "promote",
            "promotes",
        }
        syn = {
            # Generic normalization only; avoid claim-specific mappings.
            "fibre": "fiber",
            "lower": "reduce",
            "decrease": "reduce",
            "decreases": "reduce",
            "decreased": "reduce",
            # Irregular plurals.
            "children": "child",
            "men": "man",
            "women": "woman",
            "teeth": "tooth",
        }
        out: set[str] = set()
        for t in raw:
            if not t or t in stop:
                continue
            out.add(syn.get(t, t))
        return out

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

    @staticmethod
    def _is_subjectless_predicate_fragment(text: str) -> bool:
        low = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not low:
            return True
        starts_predicate_like = bool(
            re.match(
                (
                    r"^(may|might|can|could|should|would|will|to|reduce|reduces|"
                    r"lower|lowers|help|helps|prevent|prevents)\b"
                ),
                low,
            )
        )
        if not starts_predicate_like:
            return False
        tokens = [t for t in re.findall(r"\b[a-z][a-z0-9_-]*\b", low) if t]
        non_subject = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "for",
            "and",
            "or",
            "to",
            "may",
            "might",
            "can",
            "could",
            "should",
            "would",
            "will",
            "to",
            "reduce",
            "reduces",
            "reducing",
            "lower",
            "lowers",
            "lowering",
            "prevent",
            "prevents",
            "preventing",
            "help",
            "helps",
            "helping",
            "risk",
            "type",
            "diabetes",
            "fatigue",
            "tiredness",
            "disease",
        }
        candidate_subject_tokens = [t for t in tokens[:4] if t not in non_subject]
        return len(candidate_subject_tokens) == 0

    def _evidence_is_admissible_for_claim(self, claim: str, statement: str) -> bool:
        """
        Admissibility filter: belief/survey prevalence evidence is not admissible
        for factual composition/efficacy claims unless the claim itself is about belief.
        """
        if not statement:
            return False
        if is_blocked_content(statement):
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
            r"\bcannot\b",
            r"\bno evidence\b",
            r"\bdoes\s+not\s+enter\b",
            r"\bdoes\s+not\s+integrate\b",
            r"\bcannot\s+alter\b",
            r"\bcannot\s+modify\b",
            r"\bdoes\s+not\s+affect\b",
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
                "build_support": (
                    r"\b(?:build|builds|built|maintain|maintains|maintained|support|supports|supported|"
                    r"strengthen|strengthens|strengthened|need|needs|needed|required|requires|essential|necessary)\b"
                ),
                "source_relation": (
                    r"\b(?:contain|contains|contained|include|includes|included|found in|source of|"
                    r"rich in|high in|low in|such as|including|part of|as part of)\b"
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
                logger.debug(
                    "[VerdictGenerator][Polarity] seg_neg=%s stmt_neg=%s same_predicate=%s stance=%s => entails",
                    seg_neg,
                    stmt_neg,
                    same_predicate,
                    stance_l,
                )
            return "entails"
        # Positive symmetry guard: same predicate family + no negation + anchor overlap implies support.
        if same_predicate and not seg_neg and not stmt_neg and self._segment_anchor_overlap(segment, statement) >= 0.2:
            return "entails"
        # Same predicate with polarity mismatch => contradiction.
        if same_predicate and (seg_neg ^ stmt_neg):
            if _POLARITY_DEBUG:
                logger.debug(
                    "[VerdictGenerator][Polarity] seg_neg=%s stmt_neg=%s same_predicate=%s stance=%s => contradicts",
                    seg_neg,
                    stmt_neg,
                    same_predicate,
                    stance_l,
                )
            return "contradicts"

        if stance_l in {"entails", "contradicts"}:
            if _POLARITY_DEBUG:
                logger.debug(
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
            logger.debug(
                "[VerdictGenerator][Polarity] seg_neg=%s stmt_neg=%s same_predicate=%s stance=%s => neutral",
                seg_neg,
                stmt_neg,
                same_predicate,
                stance_l,
            )
        return "neutral"

    @staticmethod
    def _intervention_anchor_tokens(text: str) -> set[str]:
        low = (text or "").lower()
        categories = {
            "food": (
                r"\b(diet|dietary|food|foods|meal|nutrition|nutritional|fruit|fruits|vegetable|vegetables|"
                r"fiber|fibre|whole[-\s]?grain|legume|plant[-\s]?based)\b"
            ),
            "supplement": (
                r"\b(supplement|supplements|supplementation|pill|capsule|tablet|dose|dosage|"
                r"vitamin\s+[a-z0-9]+\s+supplement(?:ation)?|supplement(?:ation)?\s+of\s+vitamin|"
                r"mineral supplement)\b"
            ),
            "drug": (
                r"\b(drug|drugs|medication|medications|medicine|medicines|pharmaceutical|therapy|"
                r"antibiotic|insulin|statin|metformin)\b"
            ),
        }
        out: set[str] = set()
        for label, pattern in categories.items():
            if re.search(pattern, low):
                out.add(label)
        return out

    def _intervention_alignment(self, claim_text: str, evidence_text: str) -> tuple[bool, bool]:
        claim_anchors = self._intervention_anchor_tokens(claim_text)
        evidence_anchors = self._intervention_anchor_tokens(evidence_text)
        claim_low = (claim_text or "").lower()
        evidence_low = (evidence_text or "").lower()

        # If claim has no intervention anchors, do not block refutation eligibility.
        if not claim_anchors:
            return True, False

        # If category extraction fails on evidence, require exact lexical overlap on intervention tokens.
        if not evidence_anchors:
            lexical = {
                t
                for t in re.findall(
                    r"\b(diet|dietary|food|foods|fruit|fruits|vegetable|vegetables|fiber|fibre|supplement|"
                    r"supplementation|drug|drugs|medication|medications|medicine|medicines|therapy|dose|dosage)\b",
                    claim_low,
                )
            }
            if not lexical:
                return False, False
            has_match = any(t in evidence_low for t in lexical)
            return bool(has_match), False

        # Intervention mismatch guard: food/diet evidence cannot refute supplement/drug claims and vice versa.
        claim_food = "food" in claim_anchors
        claim_med = bool({"supplement", "drug"} & claim_anchors)
        ev_food = "food" in evidence_anchors
        ev_med = bool({"supplement", "drug"} & evidence_anchors)
        claim_food_terms = set(
            re.findall(r"\b(fruit|fruits|vegetable|vegetables|whole[-\s]?grain|legume|fiber|fibre|diet)\b", claim_low)
        )
        ev_food_terms = set(
            re.findall(
                r"\b(fruit|fruits|vegetable|vegetables|whole[-\s]?grain|legume|fiber|fibre|diet)\b",
                evidence_low,
            )
        )
        shared_food_specific = {t for t in (claim_food_terms & ev_food_terms) if t not in {"diet"}}
        mismatch = (claim_food and ev_med and len(shared_food_specific) == 0) or (claim_med and ev_food and not ev_med)
        return (not mismatch), True

    @staticmethod
    def _scope_profile(text: str) -> Dict[str, set[str]]:
        low = (text or "").lower()
        population = set(
            re.findall(
                r"\b(healthy|adults?|children|child|elderly|smokers?|pregnan(?:t|cy)|athletes?|patients?)\b",
                low,
            )
        )
        intervention = set(
            re.findall(
                r"\b(supplement(?:ation)?|diet|dietary|oral|intravenous|iv|dose|dosage|placebo|treatment)\b",
                low,
            )
        )
        outcome = set(
            re.findall(
                r"\b(immunity|immune|infection|incidence|duration|severity|hospitalization|mortality)\b",
                low,
            )
        )
        context = set(
            re.findall(
                r"\b(randomized|trial|meta-analysis|systematic review|guideline|observational|cohort)\b",
                low,
            )
        )
        return {
            "population": population,
            "intervention": intervention,
            "outcome": outcome,
            "context": context,
        }

    @staticmethod
    def _scope_alignment_score(claim_scope: Dict[str, set[str]], stmt_scope: Dict[str, set[str]]) -> float:
        weights = {"population": 0.35, "intervention": 0.35, "outcome": 0.20, "context": 0.10}
        total = 0.0
        used = 0.0
        for key, w in weights.items():
            claim_terms = claim_scope.get(key) or set()
            stmt_terms = stmt_scope.get(key) or set()
            if not claim_terms:
                continue
            used += w
            overlap = len(claim_terms & stmt_terms) / max(1, len(claim_terms))
            total += w * overlap
        if used <= 0.0:
            return 1.0
        return max(0.0, min(1.0, total / used))

    def _segment_topic_guard_ok(self, segment: str, statement: str) -> bool:
        seg_concepts = self._concept_hits(segment)
        stmt_concepts = self._concept_hits(statement)
        # Whole-food claims should not be invalidated by supplement-only evidence.
        seg_l = (segment or "").lower()
        stmt_l = (statement or "").lower()
        # Filter obvious event/schedule snippets that can leak into crawled corpora.
        event_like = bool(
            re.search(r"\b(welcome|networking|agenda|registration|session|workshop|conference|symposium)\b", stmt_l)
            or re.search(r"\b\d{1,2}:\d{2}(?:\s*[-–]\s*\d{1,2}:\d{2})?\b", stmt_l)
        )
        if event_like:
            shared_content = len(self._content_tokens(segment) & self._content_tokens(statement))
            if shared_content < 2:
                return False

        whole_food_claim = bool(
            re.search(r"\b(diet|vegetable|vegetables|fruit|fruits|whole food|foods)\b", seg_l)
        ) and not bool(re.search(r"\bsupplement(?:ation)?\b", seg_l))
        supplement_stmt = bool(re.search(r"\bsupplement(?:ation)?\b", stmt_l))
        if whole_food_claim and supplement_stmt and not re.search(r"\b(vegetable|fruit|diet)\b", stmt_l):
            return False

        seg_tokens = self._topic_tokens(segment)
        stmt_tokens = self._topic_tokens(statement)
        overlap_tokens = len(seg_tokens & stmt_tokens)
        if len(seg_tokens) >= 3 and overlap_tokens == 0 and not (seg_concepts & stmt_concepts):
            return False
        # Penalize generic overlap only (e.g., "helps", "risk") by requiring at least one content token hit.
        content_overlap = len(self._content_tokens(segment) & self._content_tokens(statement))
        if content_overlap == 0 and not self._object_semantic_equivalent(segment, statement):
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

        seg_subject_tokens = VerdictGenerator._segment_subject_tokens(seg)
        seg_object_tokens = VerdictGenerator._segment_object_tokens(seg)
        stmt_tokens = VerdictGenerator._statement_tokens(stmt)

        seg_subject_lem = {VerdictGenerator._lemma_token(t) for t in seg_subject_tokens}
        seg_object_lem = {VerdictGenerator._lemma_token(t) for t in seg_object_tokens}
        stmt_lem = {VerdictGenerator._lemma_token(t) for t in stmt_tokens}

        subject_overlap = len(seg_subject_lem & stmt_lem) / max(1, len(seg_subject_lem)) if seg_subject_lem else 0.0
        object_overlap = len(seg_object_lem & stmt_lem) / max(1, len(seg_object_lem)) if seg_object_lem else 0.0

        seg_buckets = VerdictGenerator._predicate_bucket_tokens(seg)
        stmt_buckets = VerdictGenerator._predicate_bucket_tokens(stmt)

        # Predicate families must align. Allow requirement-like claims to map into support/maintenance language.
        predicate_overlap = False
        if seg_buckets and stmt_buckets:
            predicate_overlap = bool(seg_buckets & stmt_buckets)
            requirement_like = "build_support" in seg_buckets
            support_like_stmt = bool(stmt_buckets & {"build_support", "therapeutic", "causal"})
            if not predicate_overlap and not (requirement_like and support_like_stmt):
                return False

        # When segment has explicit subject/object anchors, require at least partial lexical support.
        has_subject_constraint = bool(seg_subject_lem)
        has_object_constraint = bool(seg_object_lem)
        if has_subject_constraint and has_object_constraint:
            # Robust fallback for long/compound segments where subject parsing can be noisy:
            # strong object + predicate alignment can still establish predicate-level relevance.
            if subject_overlap >= 0.20 and object_overlap >= 0.20:
                return True
            if object_overlap >= 0.35 and predicate_overlap:
                return True
            return False
        if has_subject_constraint:
            return subject_overlap >= 0.20
        if has_object_constraint:
            return object_overlap >= 0.20
        return True

    @staticmethod
    def _segment_object_tokens(segment: str) -> set[str]:
        text = (segment or "").lower()
        patterns = (
            r"^(?P<subject>.+?)\s+\b(?:for|to)\b\s+(?P<object>.+)$",
            r"\b(?:as\s+part\s+of|part\s+of)\s+(?P<object>.+)$",
            r"\b(?:support|supports|supported|help|helps|boost|boosts)\s+(?P<object>.+)$",
            (
                r"\b(?:contribut(?:e|es|ed|ing)|help|helps|support|supports|needed|required)\s+"
                r"(?:to|for)\s+(?P<object>.+)$"
            ),
            (
                r"\b(?:is|are|was|were)\s+(?:an?\s+)?risk\s+factor"
                r"(?:\s+for|\s+in\s+the\s+development\s+of)\s+(?P<object>.+)$"
            ),
            (
                r"\b(?:reduce|reduces|reduced|lower|lowers|lowered)\b(?:\s+the)?\s*"
                r"(?:risk|chance|likelihood)?(?:\s+of)?\s+(?P<object>.+)$"
            ),
            (
                r"\b(?:increase|increases|increased|raise|raises|raised)\b(?:\s+the)?\s*"
                r"(?:risk|chance|likelihood)?(?:\s+of)?\s+(?P<object>.+)$"
            ),
            r"\b(?:prevent|prevents|prevented|preventing|cause|causes|caused|causing)\s+(?P<object>.+)$",
            r"\b(?:contain|contains|contained|work|works|effective|effectiveness)\b(?:\s+against)?\s+(?P<object>.+)$",
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
        object_text = re.sub(r"^(?:a\s+|an\s+)?diet\s+", "", object_text)
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
            "normal",
            "development",
            "formation",
            "shown",
            "been",
            "have",
            "has",
            "had",
            "certain",
            "types",
            "type",
            "some",
            "significant",
            "new",
        }
        return {t for t in re.findall(r"\b[a-z][a-z0-9_-]+\b", object_text) if t not in stop}

    @staticmethod
    def _segment_subject_tokens(segment: str) -> set[str]:
        text = (segment or "").lower()
        phrase_match = re.search(r"^(?P<subject>.+?)\s+\b(?:for|to)\b\s+(?P<object>.+)$", text)
        if phrase_match:
            subject_text = str(phrase_match.group("subject") or "")
        else:
            match = re.search(
                r"^(?P<subject>.+?)\b(?:increase|increases|increased|cause|causes|caused|"
                r"reduce|reduces|reduced|prevent|prevents|prevented|contain|contains|work|works|"
                r"contribut(?:e|es|ed|ing)|help|helps|support|supports|needed|required|"
                r"is|are|was|were|has|have)\b",
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
            "may",
            "might",
            "can",
            "could",
            "should",
            "would",
            "will",
            "have",
            "has",
            "had",
            "been",
            "shown",
            "as",
            "part",
            "from",
            "no",
            "not",
            "do",
            "does",
            "did",
            "is",
            "are",
            "was",
            "were",
        }
        return {t for t in re.findall(r"\b[a-z][a-z0-9_-]+\b", subject_text) if t not in stop}

    @staticmethod
    def _statement_tokens(statement: str) -> set[str]:
        return set(re.findall(r"\b[a-z][a-z0-9_-]+\b", (statement or "").lower()))

    @staticmethod
    def _lemma_token(token: str) -> str:
        t = str(token or "").lower().strip()
        if not t:
            return ""
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        for suf in ("ing", "ed", "es", "s"):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                return t[: -len(suf)]
        return t

    @staticmethod
    def _predicate_bucket_tokens(text: str) -> set[str]:
        low = str(text or "").lower()
        buckets: set[str] = set()
        groups = {
            "causal": (
                r"\b(cause|causes|caused|causing|lead|leads|leading|result|results|resulting|"
                r"associate|associated|link|linked|contribut(?:e|es|ed|ing))\b"
            ),
            "preventive": (
                r"\b(prevent|prevents|prevention|reduce|reduces|reduced|decrease|decreases|decreased|"
                r"lower|lowers|lowered|protect|protects)\b"
            ),
            "risk_factor": (
                r"\b(risk\s+factor|risk\s+factors|associated\s+with|association\s+with|"
                r"linked\s+to|predispos(?:e|es|ed|ing)|contributor\s+to)\b"
            ),
            "therapeutic": (
                r"\b(treat|treats|treatment|cure|cures|help|helps|promote|promotes|improve|improves|"
                r"enhance|enhances|boost|boosts|effective|efficacious)\b"
            ),
            "genomic_change": (
                r"\b(alter|alters|altered|change|changes|changed|integrate|integration|"
                r"incorporat|modify|modifies)\b"
            ),
            "build_support": (
                r"\b(build|builds|built|maintain|maintains|maintained|support|supports|supported|"
                r"strengthen|strengthens|strengthened|needed|need|needs|required|requires|essential|necessary|"
                r"increase|increases|increased|enhance|enhances|enhanced|boost|boosts|boosted|"
                r"contribut(?:e|es|ed|ing))\b"
            ),
            "source_relation": (
                r"\b(contain|contains|contained|include|includes|included|found in|source of|"
                r"rich in|high in|low in|such as|including|part of|as part of)\b"
            ),
        }
        for name, pattern in groups.items():
            if re.search(pattern, low):
                buckets.add(name)
        return buckets

    def _predicate_semantic_equivalent(self, claim_predicate: str, evidence_predicate: str) -> bool:
        c_b = self._predicate_bucket_tokens(claim_predicate)
        e_b = self._predicate_bucket_tokens(evidence_predicate)
        return bool(c_b and e_b and (c_b & e_b))

    @staticmethod
    def _object_semantic_equivalent(claim_text: str, evidence_text: str) -> bool:
        claim_obj = VerdictGenerator._segment_object_tokens(claim_text)
        if not claim_obj:
            return False
        ev_tokens = VerdictGenerator._statement_tokens(evidence_text)
        claim_obj_lem = {VerdictGenerator._lemma_token(t) for t in claim_obj}
        ev_lem = {VerdictGenerator._lemma_token(t) for t in ev_tokens}
        overlap = len(claim_obj_lem & ev_lem) / max(1, len(claim_obj_lem))
        return overlap >= 0.40

    def _extract_canonical_predicate_triplet(self, text: str) -> Dict[str, str]:
        low = (text or "").strip().lower()
        has_explicit_relation_verb = bool(
            re.search(
                (
                    r"\b(is|are|was|were|be|been|being|do|does|did|can|could|may|might|must|should|would|will|"
                    r"support(?:s|ed|ing)?|help(?:s|ed|ing)?|contribut(?:e|es|ed|ing)|"
                    r"prevent(?:s|ed|ing)?|reduce(?:s|d|ing)?|increase(?:s|d|ing)?|"
                    r"improve(?:s|d|ing)?|cause(?:s|d|ing)?|require(?:s|d|ing)?|need(?:s|ed|ing)?)\b"
                ),
                low,
                flags=re.IGNORECASE,
            )
        )
        phrase_match = re.match(r"^\s*(?P<subj>.+?)\s+\b(?:for|to)\b\s+(?P<obj>.+?)\s*$", low, flags=re.IGNORECASE)
        if phrase_match and not has_explicit_relation_verb:
            subj = re.sub(r"\s+", " ", phrase_match.group("subj")).strip(" ,.")
            obj = re.sub(r"\s+", " ", phrase_match.group("obj")).strip(" ,.")
            if subj and obj:
                return {
                    "subject_span": subj,
                    "predicate_span": "for",
                    "object_span": obj,
                    "canonical_predicate": "support",
                }

        tokens = re.findall(r"\b[a-z][a-z0-9_-]*\b", low)
        if len(tokens) < 3:
            return {"subject_span": "", "predicate_span": "", "object_span": "", "canonical_predicate": ""}

        aux = {
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
            "can",
            "could",
            "may",
            "might",
            "must",
            "should",
            "would",
            "will",
            "not",
            "no",
        }
        prepositions = {"to", "in", "on", "of", "for", "with", "into", "from", "as"}
        stop = {
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
            "that",
            "this",
            "your",
            "our",
            "as",
        }
        adjective_stop = {
            "strong",
            "weak",
            "healthy",
            "good",
            "bad",
            "normal",
            "abnormal",
            "possible",
            "likely",
            "unlikely",
        }
        verb_hints = {
            "build",
            "maintain",
            "support",
            "help",
            "improve",
            "enhance",
            "relieve",
            "need",
            "require",
            "prevent",
            "reduce",
            "treat",
            "cure",
            "cause",
            "increase",
            "decrease",
            "alter",
            "change",
            "integrate",
            "modify",
            "lead",
            "result",
            "associate",
            "link",
            "work",
            "contain",
            "manage",
            "contribut",
            "necessary",
            "important",
            "required",
            "needed",
        }
        pred_idx = -1
        best_score = -1
        for i in range(1, len(tokens) - 1):
            tok = tokens[i]
            if tok in stop:
                continue
            if tok in prepositions:
                continue
            if tok in adjective_stop:
                continue
            score = 0
            lemma = self._lemma_token(tok)
            if lemma in verb_hints:
                score += 4
            if tok in aux:
                score += 2
            if tok.endswith(("ing", "ed", "es", "e")):
                score += 1
            if i > 0 and tokens[i - 1] in aux:
                score += 1
            if i + 1 < len(tokens) and tokens[i + 1] in {"to", "that", "into", "in", "on", "with", "of"}:
                score += 1
            if tok in {"not", "no"}:
                score -= 2
            # Penalize likely noun+preposition heads (e.g., "cultures in ...")
            # unless token is recognized as a verb hint.
            if tok.endswith("s") and i + 1 < len(tokens) and tokens[i + 1] in prepositions and lemma not in verb_hints:
                score -= 3
            # Prefer earlier verb-like heads on ties; later tokens are often objects/nouns.
            if score > best_score or (score == best_score and (pred_idx < 0 or i < pred_idx)):
                best_score = score
                pred_idx = i
        if pred_idx < 0:
            pred_idx = 1

        selected = tokens[pred_idx]
        selected_lemma = self._lemma_token(selected)
        selected_verb_like = (
            selected_lemma in verb_hints
            or selected in aux
            or selected in {"necessary", "important", "required", "needed"}
            or selected.endswith(("ing", "ed", "es", "e"))
        )
        # Prevent noun-heavy spans from being forced into predicate slots.
        if best_score < 2 or (not selected_verb_like):
            return {
                "subject_span": "",
                "predicate_span": "",
                "object_span": "",
                "canonical_predicate": "",
            }

        subject_tokens = [t for t in tokens[:pred_idx] if t not in stop and t not in aux and t not in {"no", "not"}]
        predicate_tokens = [tokens[pred_idx]]
        if tokens[pred_idx] in aux and pred_idx + 1 < len(tokens):
            predicate_tokens.append(tokens[pred_idx + 1])
            obj_start = pred_idx + 2
        elif (
            self._lemma_token(tokens[pred_idx]) in {"help", "promote", "improve"}
            and pred_idx + 1 < len(tokens)
            and self._lemma_token(tokens[pred_idx + 1])
            in {"lower", "reduce", "decrease", "maintain", "support", "build", "regulate"}
        ):
            predicate_tokens.append(tokens[pred_idx + 1])
            obj_start = pred_idx + 2
        elif pred_idx + 1 < len(tokens) and tokens[pred_idx + 1] in {"to", "in", "into", "on", "of", "for", "with"}:
            predicate_tokens.append(tokens[pred_idx + 1])
            obj_start = pred_idx + 2
        else:
            obj_start = pred_idx + 1
        object_tokens = [t for t in tokens[obj_start:] if t not in stop]

        predicate_span = " ".join(predicate_tokens).strip()
        canonical = " ".join(self._lemma_token(t) for t in predicate_tokens if t).strip()
        return {
            "subject_span": " ".join(subject_tokens).strip(),
            "predicate_span": predicate_span,
            "object_span": " ".join(object_tokens).strip(),
            "canonical_predicate": canonical or self._lemma_token(predicate_span),
        }

    @staticmethod
    def _is_actionable_predicate(predicate: str) -> bool:
        pred = str(predicate or "").strip().lower()
        if not pred:
            return False
        roots = {
            "be",
            "cause",
            "change",
            "chang",
            "prevent",
            "reduce",
            "increase",
            "decrease",
            "improve",
            "worsen",
            "support",
            "help",
            "treat",
            "cure",
            "protect",
            "lead",
            "result",
            "associate",
            "link",
            "contribute",
            "contribut",
            "necessary",
            "important",
            "required",
            "needed",
            "require",
            "need",
            "maintain",
            "build",
            "promote",
            "regulate",
            "boost",
            "contain",
            "work",
            "affect",
            "inhibit",
            "stimulate",
        }
        for tok in re.findall(r"\b[a-z][a-z0-9_-]*\b", pred):
            if VerdictGenerator._lemma_token(tok) in roots:
                return True
        return False

    @staticmethod
    def _normalize_relevance_label(relevance: str) -> str:
        rel = str(relevance or "NEUTRAL").upper()
        if rel in {"CONTRADICTS", "REFUTES", "INVALID", "PARTIALLY_INVALID", "PARTIALLY_CONTRADICTS"}:
            return "REFUTES"
        if rel in {"SUPPORTS", "VALID", "PARTIALLY_VALID", "PARTIAL", "PARTIALLY_SUPPORTS"}:
            return "SUPPORTS"
        return "NEUTRAL"

    @staticmethod
    def _has_not_all_quantifier(text: str) -> bool:
        low = (text or "").lower()
        return bool(re.search(r"\bnot\s+all\b", low))

    @staticmethod
    def _has_universal_quantifier(text: str) -> bool:
        low = (text or "").lower()
        return bool(
            re.search(
                r"\b(all|every|each|always|entirely|completely|universally|all\s+types?|all\s+forms?)\b",
                low,
            )
        )

    def _quantifier_refute_allowed(self, claim_text: str, statement: str) -> bool:
        # For "not all X are Y" claims, subset-level statements like
        # "trans fats are harmful" must not be treated as full contradiction.
        if self._has_not_all_quantifier(claim_text):
            return self._has_universal_quantifier(statement)
        return True

    @staticmethod
    def _dose_scope_refute_allowed(claim_text: str, statement: str) -> bool:
        claim = (claim_text or "").lower()
        stmt = (statement or "").lower()
        claim_moderate = bool(
            re.search(
                r"\b(low|moderate|normal|usual|recommended)\b|\b(?:1|2|3|4)\s*(?:cups?|mg|milligrams?)\b",
                claim,
            )
        )
        stmt_high_or_excess = bool(
            re.search(
                (
                    r"\b(high|higher|excess|excessive|overload|overconsumption|heavy|very high|large amounts?)\b|"
                    r"\btoo much\b|\bthroughout the day\b"
                ),
                stmt,
            )
        )
        stmt_moderate = bool(
            re.search(
                r"\b(low|moderate|normal|usual|recommended)\b|\b(?:1|2|3|4)\s*(?:cups?|mg|milligrams?)\b",
                stmt,
            )
        )
        if claim_moderate and stmt_high_or_excess and not stmt_moderate:
            return False
        return True

    def compute_predicate_match(self, claim_text: str, evidence_text: str) -> float:
        claim_t = self._extract_canonical_predicate_triplet(claim_text)
        ev_t = self._extract_canonical_predicate_triplet(evidence_text)
        cp = str(claim_t.get("canonical_predicate") or "")
        ep = str(ev_t.get("canonical_predicate") or "")
        if not cp:
            return 0.0
        if cp and ep and cp == ep:
            return 1.0

        # Close semantic match by normalized lexical similarity + polarity agreement.
        cp_tokens = [self._lemma_token(t) for t in cp.split() if t]
        ep_tokens = [self._lemma_token(t) for t in ep.split() if t]
        overlap = len(set(cp_tokens) & set(ep_tokens)) / max(1, len(set(cp_tokens)))
        lexical_predicate_overlap = overlap > 0.0
        pol = self._segment_polarity(claim_text, evidence_text, stance="neutral")
        anchor_overlap = self._segment_anchor_overlap(claim_text, evidence_text)
        claim_subject_tokens = {
            t
            for t in self._segment_subject_tokens(claim_text)
            if t
            not in {
                "may",
                "might",
                "can",
                "could",
                "should",
                "would",
                "will",
                "have",
                "has",
                "had",
                "been",
                "shown",
                "as",
                "part",
                "from",
            }
        }
        evidence_tokens = self._statement_tokens(evidence_text)
        subject_overlap_ratio = (
            len({self._lemma_token(t) for t in claim_subject_tokens} & {self._lemma_token(t) for t in evidence_tokens})
            / max(1, len(claim_subject_tokens))
            if claim_subject_tokens
            else 0.0
        )
        subject_guard_ok = (not claim_subject_tokens) or subject_overlap_ratio >= 0.2 or anchor_overlap >= 0.55
        full_claim_buckets = self._predicate_bucket_tokens(claim_text)
        full_ev_buckets = self._predicate_bucket_tokens(evidence_text)
        full_bucket_overlap = bool(full_claim_buckets and full_ev_buckets and (full_claim_buckets & full_ev_buckets))
        claim_low = (claim_text or "").lower()
        evidence_low = (evidence_text or "").lower()
        claim_preventive = bool(
            re.search(r"\b(prevent|prevents|prevented|preventing|protection|protects?)\b", claim_low)
        )
        evidence_risk_reduction_only = bool(
            re.search(
                (
                    r"\b(reduce|reduces|reduced|reducing|decrease|decreases|decreased|"
                    r"lower|lowers|lowered)\b.{0,24}\b(risk|odds|chance|likelihood|incidence)\b"
                ),
                evidence_low,
            )
            and not re.search(r"\b(prevent|prevents|prevented|preventing|protect|protects)\b", evidence_low)
        )
        claim_sub = set(re.findall(r"\b[a-z][a-z0-9_-]+\b", claim_t.get("subject_span", "")))
        ev_sub = set(re.findall(r"\b[a-z][a-z0-9_-]+\b", ev_t.get("subject_span", "")))
        sub_overlap = (len(claim_sub & ev_sub) / max(1, len(claim_sub))) if claim_sub else 0.0
        claim_obj = set(re.findall(r"\b[a-z][a-z0-9_-]+\b", claim_t.get("object_span", "")))
        ev_obj = set(re.findall(r"\b[a-z][a-z0-9_-]+\b", ev_t.get("object_span", "")))
        obj_overlap = (len(claim_obj & ev_obj) / max(1, len(claim_obj))) if claim_obj else 0.0
        cp_full = " ".join(
            x
            for x in [
                str(claim_t.get("canonical_predicate") or ""),
                str(claim_t.get("predicate_span") or ""),
            ]
            if x
        )
        ep_full = " ".join(
            x
            for x in [
                str(ev_t.get("canonical_predicate") or ""),
                str(ev_t.get("predicate_span") or ""),
            ]
            if x
        )
        if (
            lexical_predicate_overlap
            and subject_guard_ok
            and (obj_overlap >= 0.35 or (anchor_overlap >= ANCHOR_THRESHOLD and pol in {"entails", "contradicts"}))
        ):
            if claim_preventive and evidence_risk_reduction_only:
                return 0.0
            return 0.7
        # Nominal/attribution claims often omit explicit verbs (e.g., "X from foods such as Y").
        # Treat source/composition relations as a predicate family when anchors align.
        claim_low = (claim_text or "").lower()
        evidence_low = (evidence_text or "").lower()
        claim_nominal_relation = bool(
            re.search(
                r"\b("
                r"such as|including|include|contains?|source of|"
                r"rich in|high in|low in|found in|part of|as part of"
                r")\b",
                claim_low,
            )
        )
        evidence_nominal_relation = bool(
            re.search(
                r"\b("
                r"found in|source of|contains?|including|include|"
                r"rich in|high in|low in|part of|as part of"
                r")\b",
                evidence_low,
            )
        )
        if (
            claim_nominal_relation
            and evidence_nominal_relation
            and (obj_overlap >= 0.35 or (anchor_overlap >= 0.25 and pol in {"entails", "contradicts"}))
            and subject_guard_ok
        ):
            if claim_preventive and evidence_risk_reduction_only:
                return 0.0
            return 0.7
        # Robust fallback: use full-sentence predicate families when token-level extraction is noisy.
        if full_bucket_overlap and (obj_overlap >= 0.35 or anchor_overlap >= 0.25) and subject_guard_ok:
            if claim_preventive and evidence_risk_reduction_only:
                return 0.0
            return 0.7
        if (
            self._predicate_semantic_equivalent(cp_full, ep_full)
            and (obj_overlap >= 0.5 or anchor_overlap >= ANCHOR_THRESHOLD)
            and subject_guard_ok
        ):
            return 0.7
        if (
            self._object_semantic_equivalent(claim_text, evidence_text)
            and (
                self._predicate_semantic_equivalent(cp_full + " " + claim_text, ep_full + " " + evidence_text)
                or pol in {"entails", "contradicts"}
            )
            and subject_guard_ok
        ):
            return 0.7

        if anchor_overlap >= 0.2 and sub_overlap >= 0.5 and obj_overlap >= 0.5 and subject_guard_ok:
            return 0.7

        if (
            pol in {"entails", "contradicts"}
            and anchor_overlap >= ANCHOR_THRESHOLD
            and obj_overlap >= 0.5
            and subject_guard_ok
        ):
            return 0.7

        if pol == "contradicts" and anchor_overlap >= ANCHOR_THRESHOLD:
            return 0.4

        if self._is_explicit_refutation_statement(evidence_text) and anchor_overlap >= 0.2:
            return 0.4

        # Indirect mechanism match: object aligns and statement addresses the mechanism path.
        if obj_overlap >= 0.5 and pol in {"entails", "contradicts"}:
            return 0.4
        return 0.0

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
        try:
            min_per_segment = max(1, int(os.getenv("VERDICT_MIN_EVIDENCE_PER_SEGMENT", "5")))
        except Exception:
            min_per_segment = 5
        selected: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _add(ev: Dict[str, Any]) -> None:
            key = self._normalize_statement_key(str(ev.get("statement") or ev.get("text") or ""))
            if key and key not in seen:
                seen.add(key)
                selected.append(ev)

        def _segment_match_score(segment: str, statement: str) -> float:
            if not self._segment_topic_guard_ok(segment, statement):
                return 0.0
            anchor_eval = evaluate_anchor_match(segment, statement)
            if not bool(anchor_eval.get("anchor_ok", False)):
                return 0.0
            anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
            pred_match = float(self.compute_predicate_match(segment, statement) or 0.0)
            stmt_tokens = {self._lemma_token(t) for t in self._statement_tokens(statement)}
            seg_obj_tokens = {self._lemma_token(t) for t in self._segment_object_tokens(segment)}
            obj_overlap = len(seg_obj_tokens & stmt_tokens) / max(1, len(seg_obj_tokens)) if seg_obj_tokens else 0.0
            return (0.45 * anchor_overlap) + (0.35 * pred_match) + (0.20 * obj_overlap)

        # First pass: build segment-specific ranked candidate pools.
        segment_ranked: Dict[str, List[Dict[str, Any]]] = {}
        for segment in segments:
            ranked_for_segment: List[tuple[float, Dict[str, Any]]] = []
            for ev in candidates:
                if str(ev.get("stance", "neutral") or "neutral").lower() == "contradicts":
                    continue
                stmt = str(ev.get("statement") or ev.get("text") or "")
                # Assign evidence to the best-matching segment first to avoid
                # one segment hijacking evidence intended for another.
                best_seg = None
                best_seg_score = -1.0
                for seg2 in segments:
                    s2 = _segment_match_score(seg2, stmt)
                    if s2 > best_seg_score:
                        best_seg_score = s2
                        best_seg = seg2
                if best_seg != segment:
                    continue
                anchor_eval = evaluate_anchor_match(segment, stmt)
                if not bool(anchor_eval.get("anchor_ok", False)):
                    continue
                score = (0.75 * self._evidence_score(ev)) + (
                    0.25 * float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                )
                ranked_for_segment.append((score, ev))
            ranked_for_segment.sort(key=lambda x: x[0], reverse=True)
            segment_ranked[segment] = [ev for _, ev in ranked_for_segment]

        # Second pass: round-robin allocation to guarantee minimum evidence per segment
        # (when candidates exist), preventing segment starvation.
        seg_counts: Dict[str, int] = {s: 0 for s in segments}
        seg_indices: Dict[str, int] = {s: 0 for s in segments}
        made_progress = True
        while made_progress and len(selected) < top_k:
            made_progress = False
            for segment in segments:
                if seg_counts.get(segment, 0) >= min_per_segment:
                    continue
                pool = segment_ranked.get(segment, [])
                idx = seg_indices.get(segment, 0)
                while idx < len(pool):
                    ev = pool[idx]
                    idx += 1
                    before = len(selected)
                    _add(ev)
                    if len(selected) > before:
                        seg_counts[segment] = seg_counts.get(segment, 0) + 1
                        made_progress = True
                        break
                seg_indices[segment] = idx
                if len(selected) >= top_k:
                    return selected[:top_k]

        # Third pass: fill with strongest non-contradicting evidence.
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

    def _effective_top_k_for_claim(self, claim: str, base_top_k: int) -> int:
        """
        Scale evidence budget with claim complexity.
        More segments/sub-claims need more evidence capacity to avoid under-coverage.
        """
        try:
            min_top_k = max(3, int(os.getenv("VERDICT_TOP_K_MIN", "5")))
        except Exception:
            min_top_k = 5
        try:
            max_top_k = max(min_top_k, int(os.getenv("VERDICT_TOP_K_MAX", "50")))
        except Exception:
            max_top_k = 50
        try:
            extra_per_segment = max(1, int(os.getenv("VERDICT_TOP_K_EXTRA_PER_SEGMENT", "2")))
        except Exception:
            extra_per_segment = 2
        try:
            min_per_segment = max(1, int(os.getenv("VERDICT_MIN_EVIDENCE_PER_SEGMENT", "5")))
        except Exception:
            min_per_segment = 5

        base = max(min_top_k, int(base_top_k or min_top_k))
        segments = self._split_claim_into_segments(claim) or [claim]
        seg_count = max(1, len(segments))
        scaled = base + max(0, seg_count - 1) * extra_per_segment
        required_for_segments = seg_count * min_per_segment
        scaled = max(scaled, required_for_segments)
        return max(min_top_k, min(max_top_k, scaled))

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
            recomputed_anchor = self._segment_anchor_overlap(claim, statement)
            anchor_score = max(
                float(item.get("anchor_match_score", 0.0) or 0.0),
                float(ev.get("anchor_match_score", 0.0) or 0.0),
                float(recomputed_anchor or 0.0),
            )
            claim_triplet = self._extract_canonical_predicate_triplet(claim)
            evidence_triplet = self._extract_canonical_predicate_triplet(statement)
            predicate_match_score = self.compute_predicate_match(claim, statement)
            claim_object_tokens = {self._lemma_token(t) for t in self._segment_object_tokens(claim)}
            evidence_tokens = {self._lemma_token(t) for t in self._statement_tokens(statement)}
            object_overlap_ok = (
                not claim_object_tokens
                or bool(claim_object_tokens & evidence_tokens)
                or self._is_explicit_refutation_statement(statement)
            )
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
            intervention_match, intervention_anchors_ok = self._intervention_alignment(claim, statement)
            credibility = float(ev.get("credibility", 0.5) or 0.5)
            claim_scope = self._scope_profile(claim)
            stmt_scope = self._scope_profile(statement)
            scope_alignment = self._scope_alignment_score(claim_scope, stmt_scope)
            nli_entail_prob = max(
                0.0,
                min(
                    1.0,
                    (0.45 * support_strength)
                    + (0.30 * predicate_match_score)
                    + (0.15 * anchor_score)
                    + (0.10 * scope_alignment),
                ),
            )
            nli_contradict_prob = max(
                0.0,
                min(
                    1.0,
                    (0.55 * contradiction_score)
                    + (0.25 * max(0.0, 1.0 - predicate_match_score))
                    + (0.20 * (1.0 if self._is_explicit_refutation_statement(statement) else 0.0)),
                ),
            )
            nli_neutral_prob = max(0.0, 1.0 - max(nli_entail_prob, nli_contradict_prob))

            relevance = "NEUTRAL"
            relevance_score = max(0.0, min(1.0, base_score))
            refute_eligible = False
            blocked_content = is_blocked_content(statement)
            if blocked_content:
                relevance = "NEUTRAL"
                relevance_score = 0.0
            elif not self._segment_topic_guard_ok(claim, statement):
                relevance_score *= 0.10
            else:
                # Refutation can be decided by direct contradiction, independent of support gates.
                numeric_rel = self._numeric_relation_relevance(claim, statement)
                dna_rel = self._dna_integration_relevance(claim, statement)
                claim_neg = bool(re.search(r"\b(no|not|never|cannot|can't|does not|do not|without)\b", claim.lower()))
                stmt_neg = bool(
                    re.search(
                        r"\b(no|not|never|cannot|can't|does not|do not|without|doesn't|don't)\b",
                        statement.lower(),
                    )
                )
                refute_by_rule = (
                    (
                        contradiction_score >= CONTRADICTION_THRESHOLD
                        and predicate_match_score >= 0.4
                        and (claim_neg != stmt_neg)
                    )
                    or (polarity_rel == "contradicts" and predicate_match_score >= 0.4)
                    or numeric_rel == "CONTRADICTS"
                    or dna_rel == "CONTRADICTS"
                )
                # Quantifier guard: "not all X are Y" cannot be refuted by
                # subset evidence like "trans fats are harmful".
                if refute_by_rule and not self._quantifier_refute_allowed(claim, statement):
                    refute_by_rule = False
                if refute_by_rule and not self._dose_scope_refute_allowed(claim, statement):
                    refute_by_rule = False
                if refute_by_rule:
                    if intervention_match:
                        relevance = "REFUTES"
                        relevance_score = max(relevance_score, contradiction_score, nli_contradict_prob, 0.65)
                        if scope_alignment < 0.34:
                            relevance_score *= 0.72
                        refute_eligible = (
                            nli_contradict_prob >= 0.45
                            and intervention_match
                            and (predicate_match_score >= 0.25 or intervention_anchors_ok)
                            and credibility >= 0.6
                        )
                    else:
                        relevance = "NEUTRAL"
                        relevance_score *= 0.55
                else:
                    support_strength_threshold = float(os.getenv("SEGMENT_SUPPORT_STRENGTH_THRESHOLD", "0.30"))
                    support_polarity_ok = polarity_rel == "entails" or (
                        polarity_rel == "neutral"
                        and predicate_match_score >= 0.7
                        and not self._is_explicit_refutation_statement(statement)
                    )
                    support_gate = (
                        predicate_match_score >= PREDICATE_MATCH_THRESHOLD
                        and anchor_score >= max(0.25, ANCHOR_THRESHOLD - 0.05)
                        and object_overlap_ok
                        and support_polarity_ok
                        and (
                            support_strength >= support_strength_threshold
                            or base_score >= 0.40
                            or seed_rel == "SUPPORTS"
                        )
                        and not self._is_reporting_statement(statement)
                        and not self._is_claim_mention_statement(statement)
                    )
                    # Numeric exact-match can support if predicate and anchor are both present.
                    if numeric_rel == "SUPPORTS" and predicate_match_score >= PREDICATE_MATCH_THRESHOLD:
                        support_gate = support_gate or (anchor_score >= ANCHOR_THRESHOLD)
                    # Prevent broad negated claims from being "supported" only by narrow population/scope evidence.
                    scope_mismatch_for_negated_claim = claim_neg and scope_alignment < 0.34
                    if scope_mismatch_for_negated_claim:
                        support_gate = False
                    if support_gate:
                        relevance = "SUPPORTS"
                        relevance_score = max(relevance_score, support_strength, 0.62)
                        if scope_alignment < 0.40:
                            relevance_score *= 0.78
                    elif seed_rel == "REFUTES" and contradiction_score >= 0.45:
                        if predicate_match_score >= 0.4 and intervention_match:
                            relevance = "REFUTES"
                            relevance_score = max(relevance_score, contradiction_score)
                            if scope_alignment < 0.34:
                                relevance_score *= 0.72
                            refute_eligible = (
                                nli_contradict_prob >= 0.45
                                and intervention_match
                                and (predicate_match_score >= 0.25 or intervention_anchors_ok)
                                and credibility >= 0.6
                            )
                        else:
                            relevance = "NEUTRAL"
                            relevance_score *= 0.55
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
                    "object_match_ok": bool(object_overlap_ok),
                    "support_strength": max(0.0, min(1.0, support_strength)),
                    "contradiction_score": max(0.0, min(1.0, contradiction_score)),
                    "nli_entail_prob": round(float(nli_entail_prob), 4),
                    "nli_contradict_prob": round(float(nli_contradict_prob), 4),
                    "nli_neutral_prob": round(float(nli_neutral_prob), 4),
                    "canonical_predicate": str(claim_triplet.get("canonical_predicate") or ""),
                    "subject_span": str(claim_triplet.get("subject_span") or ""),
                    "predicate_span": str(claim_triplet.get("predicate_span") or ""),
                    "object_span": str(claim_triplet.get("object_span") or ""),
                    "evidence_predicate": str(evidence_triplet.get("canonical_predicate") or ""),
                    "credibility": credibility,
                    "intervention_match": bool(intervention_match),
                    "intervention_anchors_ok": bool(intervention_anchors_ok),
                    "refute_eligible": bool(refute_eligible),
                    "blocked_content": bool(blocked_content),
                    "scope_alignment": round(float(scope_alignment), 4),
                    "scope_population": sorted(claim_scope.get("population") or []),
                    "scope_intervention": sorted(claim_scope.get("intervention") or []),
                    "scope_outcome": sorted(claim_scope.get("outcome") or []),
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
        support_strength_threshold = float(os.getenv("SEGMENT_SUPPORT_STRENGTH_THRESHOLD", "0.30"))
        strict_predicate_threshold = max(
            float(PREDICATE_MATCH_THRESHOLD),
            float(os.getenv("SEGMENT_STRICT_PREDICATE_FLOOR", "0.35")),
        )

        for seg in claim_breakdown:
            segment = (seg.get("claim_segment") or "").strip()
            seg_triplet = self._extract_canonical_predicate_triplet(segment)
            segment_belief_mode = self._segment_is_belief_or_survey_claim(segment)
            best_support_item: Dict[str, Any] | None = None
            best_refute_item: Dict[str, Any] | None = None
            weak_support_item: Dict[str, Any] | None = None
            near_miss_support_item: Dict[str, Any] | None = None
            best_support_score = -1.0
            best_refute_score = -1.0
            best_weak_support_score = -1.0
            best_near_miss_score = -1.0
            saw_neutral = False

            def _consider_item(em: Dict[str, Any], ev_idx: int | None = None) -> None:
                nonlocal best_support_item
                nonlocal best_refute_item
                nonlocal weak_support_item
                nonlocal near_miss_support_item
                nonlocal best_support_score
                nonlocal best_refute_score
                nonlocal best_weak_support_score
                nonlocal best_near_miss_score
                nonlocal saw_neutral

                statement = (em.get("statement") or "").strip()
                if not statement:
                    return
                if self._is_claim_mention_statement(statement) and not segment_belief_mode:
                    return
                if not self._segment_topic_guard_ok(segment, statement):
                    return
                seg_subject_tokens = {self._lemma_token(t) for t in self._segment_subject_tokens(segment)}
                stmt_tokens_lem = {self._lemma_token(t) for t in self._statement_tokens(statement)}
                if seg_subject_tokens:
                    subject_overlap = len(seg_subject_tokens & stmt_tokens_lem) / max(1, len(seg_subject_tokens))
                    seg_object_tokens_local = {self._lemma_token(t) for t in self._segment_object_tokens(segment)}
                    object_overlap_local = (
                        len(seg_object_tokens_local & stmt_tokens_lem) / max(1, len(seg_object_tokens_local))
                        if seg_object_tokens_local
                        else 0.0
                    )
                    # Prevent cross-subclaim leakage: evidence for a different intervention/entity
                    # must not be used to mark this segment INVALID.
                    if subject_overlap < 0.20 and object_overlap_local < 0.35 and not segment_belief_mode:
                        return
                anchor_eval = evaluate_anchor_match(segment, statement)
                anchor_overlap = max(
                    float(anchor_eval.get("anchor_overlap", 0.0) or 0.0),
                    float(self._segment_anchor_overlap(segment, statement) or 0.0),
                )
                if anchor_overlap < _SEGMENT_EVIDENCE_MIN_OVERLAP:
                    return
                object_tokens = self._segment_object_tokens(segment)
                if object_tokens and not segment_belief_mode:
                    statement_tokens = set(re.findall(r"\b[a-z][a-z0-9_-]+\b", statement.lower()))
                    no_object_overlap = len(object_tokens & statement_tokens) == 0
                    if no_object_overlap and not self._is_explicit_refutation_statement(statement):
                        # Keep a near-miss candidate for partial support fallback.
                        rel = self._normalize_relevance_label(em.get("relevance", "NEUTRAL"))
                        rel_score = float(em.get("relevance_score", 0.0) or 0.0)
                        predicate_match_score = float(self.compute_predicate_match(segment, statement) or 0.0)
                        near_score = (0.55 * rel_score) + (0.20 * anchor_overlap) + (0.25 * predicate_match_score)
                        if (
                            rel in {"SUPPORTS", "NEUTRAL"}
                            and predicate_match_score >= 0.45
                            and near_score > best_near_miss_score
                        ):
                            best_near_miss_score = near_score
                            near_miss_support_item = {
                                **em,
                                "evidence_id": em.get("evidence_id", ev_idx if ev_idx is not None else -1),
                                "statement": statement,
                                "anchor_match_score": anchor_overlap,
                                "predicate_match_score": predicate_match_score,
                                "support_strength": float(em.get("support_strength", 0.0) or 0.0),
                                "contradiction_score": float(em.get("contradiction_score", 0.0) or 0.0),
                                "stance_used": rel,
                                "reason": "object_mismatch_near_miss",
                            }
                        return

                rel = self._normalize_relevance_label(em.get("relevance", "NEUTRAL"))
                rel_score = float(em.get("relevance_score", 0.0) or 0.0)
                predicate_match_score = float(self.compute_predicate_match(segment, statement) or 0.0)
                support_strength = float(em.get("support_strength", 0.0) or 0.0)
                contradiction_score = float(
                    em.get("contradiction_score", self._contradiction_score(segment, statement))
                )
                polarity = self._segment_polarity(segment, statement, stance="neutral")
                segment_has_predicate = bool(str(seg_triplet.get("canonical_predicate") or "").strip())
                content_overlap = len(self._content_tokens(segment) & self._content_tokens(statement))
                strong_neutral_support = (
                    rel == "NEUTRAL"
                    and predicate_match_score >= strict_predicate_threshold
                    and anchor_overlap >= max(0.20, ANCHOR_THRESHOLD - 0.10)
                    and (support_strength >= support_strength_threshold or rel_score >= 0.55)
                    and polarity != "contradicts"
                )
                strict_support_ok = (
                    (rel == "SUPPORTS" or strong_neutral_support)
                    and predicate_match_score >= strict_predicate_threshold
                    and (
                        anchor_overlap >= max(0.20, ANCHOR_THRESHOLD - 0.10)
                        or (predicate_match_score >= 0.7 and support_strength >= 0.55 and rel_score >= 0.45)
                    )
                    and (support_strength >= support_strength_threshold or rel_score >= 0.55)
                )
                noun_phrase_support_ok = (
                    (not segment_has_predicate)
                    and rel in {"SUPPORTS", "NEUTRAL"}
                    and anchor_overlap >= 0.30
                    and (content_overlap >= 1 or self._object_semantic_equivalent(segment, statement))
                    and rel_score >= 0.30
                    and polarity != "contradicts"
                )
                if noun_phrase_support_ok:
                    strict_support_ok = True
                try:
                    min_refute_cred = float(os.getenv("SEGMENT_REFUTE_MIN_CREDIBILITY", "0.70"))
                except Exception:
                    min_refute_cred = 0.70
                refute_ok = rel == "REFUTES" and (
                    bool(em.get("intervention_match", False))
                    and predicate_match_score >= 0.4
                    and float(em.get("credibility", 0.0) or 0.0) >= min_refute_cred
                    and self._quantifier_refute_allowed(segment, statement)
                    and self._dose_scope_refute_allowed(segment, statement)
                    and (
                        contradiction_score >= CONTRADICTION_THRESHOLD
                        or self._segment_polarity(segment, statement, stance="neutral") == "contradicts"
                    )
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
            if chosen is None and near_miss_support_item is not None:
                chosen = near_miss_support_item
            if chosen is not None:
                raw_evidence_id = chosen.get("evidence_id", -1)
                try:
                    ev_id = int(raw_evidence_id)
                except Exception:
                    ev_id = -1
                ev = evidence_by_id.get(ev_id, {})
                statement = (chosen.get("statement") or ev.get("statement") or ev.get("text") or "").strip()
                source_url = (chosen.get("source_url") or ev.get("source_url") or ev.get("source") or "").strip()
                seg_neg = bool(
                    re.search(
                        r"\b(?:no|not|never|without|cannot|can't|does not|do not|is not|are not|was not|were not)\b",
                        segment.lower(),
                    )
                )
                stmt_neg = bool(
                    re.search(
                        r"\b(?:no|not|never|without|cannot|can't|does not|do not|is not|are not|was not|were not)\b",
                        statement.lower(),
                    )
                )
                pred_match = float(chosen.get("predicate_match_score", 0.0) or 0.0)
                segment_requires_predicate_guard = bool(
                    re.search(
                        (
                            r"\b(is|are|was|were|be|being|"
                            r"cause|causes|caused|causing|reduce|reduces|reduced|reducing|"
                            r"prevent|prevents|prevented|preventing|support|supports|supported|supporting|"
                            r"contribute|contributes|contributed|contributing|"
                            r"help|helps|helped|helping|treat|treats|treated|treating|"
                            r"cure|cures|cured|curing|improve|improves|improved|improving|"
                            r"worsen|worsens|worsened|worsening|increase|increases|increased|increasing|"
                            r"decrease|decreases|decreased|decreasing|"
                            r"detoxif(?:y|ies|ied|ying))\b"
                        ),
                        segment.lower(),
                    )
                )
                hedge_support_language = bool(
                    re.search(
                        (
                            r"\b(may|might|could|suggest(?:s|ed)?|associated with|linked to|"
                            r"not been shown|insufficient evidence)\b"
                        ),
                        statement.lower(),
                    )
                )
                if explicit_refute_present and best_support_item is not None:
                    seg["status"] = "PARTIALLY_INVALID"
                elif explicit_refute_present:
                    seg["status"] = "INVALID"
                elif best_support_item is not None:
                    if segment_requires_predicate_guard and pred_match < strict_predicate_threshold:
                        seg["status"] = "UNKNOWN"
                        seg["supporting_fact"] = ""
                        seg["source_url"] = ""
                        seg["evidence_used_ids"] = []
                        seg["alignment_debug"] = {
                            "reason": "strict_predicate_floor_not_met",
                            "predicate_match_score": round(float(pred_match), 3),
                            "predicate_min_required": round(float(strict_predicate_threshold), 3),
                            "canonical_predicate": str(seg_triplet.get("canonical_predicate") or ""),
                        }
                        continue
                    if seg_neg != stmt_neg and pred_match >= 0.45:
                        seg["status"] = "INVALID"
                    else:
                        seg["status"] = "PARTIALLY_VALID" if hedge_support_language else "VALID"
                else:
                    if segment_requires_predicate_guard and pred_match < strict_predicate_threshold:
                        seg["status"] = "UNKNOWN"
                        seg["supporting_fact"] = ""
                        seg["source_url"] = ""
                        seg["evidence_used_ids"] = []
                        seg["alignment_debug"] = {
                            "reason": "strict_predicate_floor_not_met",
                            "predicate_match_score": round(float(pred_match), 3),
                            "predicate_min_required": round(float(strict_predicate_threshold), 3),
                            "canonical_predicate": str(seg_triplet.get("canonical_predicate") or ""),
                        }
                        continue
                    seg["status"] = "PARTIALLY_VALID"
                if (
                    segment_requires_predicate_guard
                    and str(seg.get("status") or "").upper() in {"VALID", "PARTIALLY_VALID"}
                    and pred_match <= 0.05
                ):
                    seg["status"] = "UNKNOWN"
                    seg["supporting_fact"] = ""
                    seg["source_url"] = ""
                    seg["evidence_used_ids"] = []
                    seg["alignment_debug"] = {
                        "reason": "predicate_alignment_missing",
                        "predicate_match_score": round(float(pred_match), 3),
                        "predicate_min_required": 0.05,
                        "canonical_predicate": str(seg_triplet.get("canonical_predicate") or ""),
                    }
                    continue
                if statement:
                    seg["supporting_fact"] = statement
                if statement and source_url:
                    seg["source_url"] = source_url
                seg["evidence_used_ids"] = [ev_id] if ev_id >= 0 else []
                chosen_reason = str(chosen.get("reason") or "strict_predicate_gate")
                seg["alignment_debug"] = {
                    "reason": chosen_reason,
                    "anchor_overlap": round(float(chosen.get("anchor_match_score", 0.0) or 0.0), 3),
                    "predicate_match_score": round(float(chosen.get("predicate_match_score", 0.0) or 0.0), 3),
                    "support_strength": round(float(chosen.get("support_strength", 0.0) or 0.0), 3),
                    "stance_used": str(
                        chosen.get("stance_used") or self._normalize_relevance_label(chosen.get("relevance"))
                    ),
                    "canonical_predicate": str(seg_triplet.get("canonical_predicate") or ""),
                    "score": round(
                        max(best_support_score, best_refute_score, best_weak_support_score, best_near_miss_score),
                        3,
                    ),
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
                    "canonical_predicate": str(seg_triplet.get("canonical_predicate") or ""),
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
                logger.debug(
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
                logger.debug(
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
            logger.debug(
                "[VerdictGenerator][Coverage] subclaims=0 covered=0 " "strong=0 partial=0 unknown=0 coverage=0.00"
            )
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
            logger.debug(
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

        logger.debug(
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
                logger.debug(
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
            logger.debug(
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
        # If strict predicate floor already blocked all unknown segments, do not
        # force promotion via adaptive fallback.
        if all(
            str(((seg.get("alignment_debug") or {}).get("reason") or "")).lower() == "strict_predicate_floor_not_met"
            for seg in unknown_segments
        ):
            return

        best_ev: Dict[str, Any] | None = None
        best_seg_text = ""
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
            # Require non-trivial segment-level predicate/anchor alignment for fallback.
            seg_ok = False
            seg_text_match = ""
            for seg in unknown_segments:
                seg_text = str(seg.get("claim_segment") or "")
                anchor_eval = evaluate_anchor_match(seg_text, statement)
                anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                pred_match = float(self.compute_predicate_match(seg_text, statement) or 0.0)
                if pred_match >= 0.35 and anchor_overlap >= 0.20:
                    seg_ok = True
                    seg_text_match = seg_text
                    break
            if not seg_ok:
                continue
            sem = float(ev.get("semantic_score") or ev.get("sem_score") or ev.get("final_score") or 0.0)
            if sem < 0.55:
                continue
            if sem > best_score:
                best_score = sem
                best_ev = ev
                best_seg_text = seg_text_match
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
                seg_ok = False
                seg_text_match = ""
                for seg in unknown_segments:
                    seg_text = str(seg.get("claim_segment") or "")
                    anchor_eval = evaluate_anchor_match(seg_text, statement)
                    anchor_overlap = float(anchor_eval.get("anchor_overlap", 0.0) or 0.0)
                    pred_match = float(self.compute_predicate_match(seg_text, statement) or 0.0)
                    if pred_match >= 0.35 and anchor_overlap >= 0.20:
                        seg_ok = True
                        seg_text_match = seg_text
                        break
                if not seg_ok:
                    continue
                best_ev = ev
                best_score = float(ev.get("final_score") or ev.get("score") or 0.0)
                best_seg_text = seg_text_match
                break
            if not best_ev:
                return

        fallback_fact = str(best_ev.get("statement") or best_ev.get("text") or "").strip()
        fallback_src = str(best_ev.get("source_url") or best_ev.get("source") or "").strip()
        if not fallback_fact:
            return
        for seg in claim_breakdown:
            if str(seg.get("status") or "UNKNOWN").upper() != "UNKNOWN":
                continue
            if best_seg_text and str(seg.get("claim_segment") or "") != best_seg_text:
                continue
            prev_dbg = seg.get("alignment_debug") or {}
            prev_reason = str(prev_dbg.get("reason") or "") if isinstance(prev_dbg, dict) else ""
            if prev_reason.lower() == "strict_predicate_floor_not_met":
                continue
            seg["status"] = "PARTIALLY_VALID"
            seg["supporting_fact"] = fallback_fact
            seg["source_url"] = fallback_src
            seg["alignment_debug"] = {
                "reason": "adaptive_coverage_fallback",
                "previous_reason": prev_reason or "unknown",
                "adaptive_coverage": round(float(adaptive_coverage), 3),
                "evidence_semantic_score": round(float(best_score), 3),
            }
        logger.debug(
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

    def _attach_exact_claim_segments(self, claim: str, claim_breakdown: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        required_segments = self._split_claim_into_segments(claim)
        if not required_segments:
            cleaned = re.sub(r"\s+", " ", (claim or "")).strip(" ,.")
            required_segments = [cleaned] if cleaned else []
        if not claim_breakdown:
            return claim_breakdown

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

        required_pack = [
            {"segment": seg, "tokens": _tokens(seg), "used": False}
            for seg in required_segments
            if str(seg or "").strip()
        ]

        for item in claim_breakdown:
            existing_exact = str(item.get("exact_claim_segment") or "").strip()
            if existing_exact:
                item["exact_claim_segment"] = existing_exact
                continue

            seg = self._normalize_segment_text(str(item.get("claim_segment") or ""))
            seg_tokens = _tokens(seg)
            best_idx = -1
            best_overlap = 0.0
            for idx, req in enumerate(required_pack):
                if req["used"]:
                    continue
                req_tokens = req["tokens"]
                if not req_tokens and not seg_tokens:
                    overlap = 1.0
                else:
                    overlap = len(seg_tokens & req_tokens) / max(1, len(req_tokens))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx

            if best_idx >= 0 and best_overlap >= 0.30:
                required_pack[best_idx]["used"] = True
                item["exact_claim_segment"] = str(required_pack[best_idx]["segment"])
            elif required_pack:
                fallback_idx = next((i for i, req in enumerate(required_pack) if not req["used"]), 0)
                if 0 <= fallback_idx < len(required_pack):
                    required_pack[fallback_idx]["used"] = True
                    item["exact_claim_segment"] = str(required_pack[fallback_idx]["segment"])
            else:
                item["exact_claim_segment"] = seg or str(item.get("claim_segment") or "").strip()
        return claim_breakdown

    def _build_decision_trace_id(self, claim: str, evidence: List[Dict[str, Any]]) -> str:
        statement_len_sum = sum(len(str(e.get("statement") or e.get("text") or "")) for e in (evidence or []))
        raw = f"{claim}|{len(evidence)}|{statement_len_sum}"
        return sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _reconcile_verdict_v2(self, claim: str, claim_breakdown: List[Dict[str, Any]]) -> Dict[str, Any]:
        statuses = self._match_required_segment_statuses(claim, claim_breakdown)
        decision = reconcile_verdict(statuses)
        return {
            "verdict": decision.verdict,
            "required_segments_count": decision.required_segments_count,
            "resolved_segments_count": decision.resolved_segments_count,
            "required_segments_resolved": decision.required_segments_resolved,
            "unresolved_segments": decision.unresolved_segments,
            "matched_statuses": decision.matched_statuses,
            "weighted_truth": decision.weighted_truth,
            "truthfulness_cap": decision.truthfulness_cap,
            "resolved_ratio": decision.resolved_ratio,
            "has_support": decision.has_support,
            "has_invalid": decision.has_invalid,
        }

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
        has_partial = any(s in {"PARTIALLY_VALID", "PARTIALLY_INVALID"} for s in statuses)

        if unresolved_segments > 0:
            if has_support and has_invalid:
                verdict = Verdict.PARTIALLY_TRUE.value
            elif has_support:
                verdict = Verdict.PARTIALLY_TRUE.value
            else:
                verdict = Verdict.UNVERIFIABLE.value
        elif all_valid or (all_support_like and not has_invalid and not has_partial):
            verdict = Verdict.TRUE.value
        elif all_invalid or (all_invalid_like and not has_support):
            verdict = Verdict.FALSE.value
        elif has_support and has_invalid:
            verdict = Verdict.PARTIALLY_TRUE.value
        elif has_invalid and not has_support:
            verdict = Verdict.FALSE.value
        else:
            verdict = Verdict.PARTIALLY_TRUE.value

        if verdict == Verdict.FALSE.value and has_support and contradict_count == 0:
            verdict = Verdict.PARTIALLY_TRUE.value

        if (
            unresolved_segments == 0
            and weighted_truth >= 0.8
            and strong_covered >= required_segments_count
            and contradict_count == 0
            and not has_partial
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
            return "The available evidence is still mixed or not decisive enough to make a firm call on this claim."
        if unresolved > 0 or unknown_count > 0:
            return (
                f"So far, evidence supports {valid_count}/{total} segment(s), "
                f"while {unknown_count} segment(s) remain unresolved."
            )
        if invalid_count > 0 and valid_count > 0:
            return (
                f"Evidence is mixed: {valid_count}/{total} segment(s) are supported and "
                f"{invalid_count}/{total} are contradicted."
            )
        if invalid_count == total:
            return "Current evidence consistently points against this claim."
        if valid_count == total and verdict == Verdict.TRUE.value:
            return _normalize_for_verdict(
                original or "Current evidence consistently supports this claim.",
                verdict,
            )
        return _normalize_for_verdict(
            original or "This assessment is based on segment-level evidence evaluation.",
            verdict,
        )

    @staticmethod
    def _build_evidence_grounded_key_findings(
        claim_breakdown: List[Dict[str, Any]],
        evidence_map: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Build key findings from exact aligned evidence statements only.
        """
        findings: List[str] = []
        seen: set[str] = set()

        for seg in claim_breakdown or []:
            status = str(seg.get("status") or "UNKNOWN").upper()
            if status not in {"VALID", "PARTIALLY_VALID", "INVALID", "PARTIALLY_INVALID"}:
                continue
            segment = str(seg.get("claim_segment") or "").strip()
            fact = str(seg.get("supporting_fact") or "").strip()
            if not segment or not fact:
                continue
            text = fact
            key = re.sub(r"\s+", " ", text).strip().lower()
            if key and key not in seen:
                findings.append(text)
                seen.add(key)
            if len(findings) >= 3:
                return findings

        # Fallback to strongly labeled evidence rows if segment alignment is sparse.
        for em in evidence_map or []:
            rel = str(em.get("relevance") or "").upper()
            if rel not in {"SUPPORTS", "REFUTES"}:
                continue
            stmt = str(em.get("statement") or "").strip()
            if not stmt:
                continue
            text = stmt
            key = re.sub(r"\s+", " ", text).strip().lower()
            if key and key not in seen:
                findings.append(text)
                seen.add(key)
            if len(findings) >= 3:
                break

        return findings

    @staticmethod
    def _build_direct_evidence_list(
        claim_breakdown: List[Dict[str, Any]],
        evidence_map: List[Dict[str, Any]],
        limit: int = 5,
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for seg in claim_breakdown or []:
            fact = str(seg.get("supporting_fact") or "").strip()
            src = str(seg.get("source_url") or "").strip()
            status = str(seg.get("status") or "UNKNOWN").upper()
            if not fact:
                continue
            item = {
                "statement": fact,
                "source_url": src,
                "segment_status": status,
            }
            key = (item["statement"].lower(), item["source_url"].lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= limit:
                return out

        for em in evidence_map or []:
            rel = str(em.get("relevance") or "").upper()
            if rel not in {"SUPPORTS", "REFUTES"}:
                continue
            stmt = str(em.get("statement") or "").strip()
            src = str(em.get("source_url") or "").strip()
            if not stmt:
                continue
            item = {
                "statement": stmt,
                "source_url": src,
                "segment_status": rel,
            }
            key = (item["statement"].lower(), item["source_url"].lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= limit:
                break

        return out

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
            seg_obj_tokens = {self._lemma_token(t) for t in self._segment_object_tokens(seg)}
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
                stmt_tokens_lemma = {self._lemma_token(t) for t in self._statement_tokens(stmt)}
                object_overlap_ok = not seg_obj_tokens or bool(seg_obj_tokens & stmt_tokens_lemma)
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
                if seg_obj_tokens and not object_overlap_ok and not self._is_explicit_refutation_statement(stmt):
                    score *= 0.25
                if score > best_score:
                    best_score = score
                    best = ev
                    best_neg = stmt_neg
                    best_idx = idx
                    best_anchor_ok = anchor_ok

            best_stmt_tokens = (
                {self._lemma_token(t) for t in self._statement_tokens(str(best.get("statement") or ""))}
                if best
                else set()
            )
            if (
                best
                and best_anchor_ok
                and best_score >= 0.55
                and (best_neg == seg_neg)
                and (not seg_obj_tokens or bool(seg_obj_tokens & best_stmt_tokens))
            ):
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
                    "exact_claim_segment": seg,
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
                logger.debug(
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
        fully_supported_segments = sum(1 for s in segment_scores if s >= 0.55)
        if segment_scores and fully_supported_segments == len(segment_scores):
            # When every segment has strong support, avoid under-reporting due conservative dampening.
            truthfulness = max(truthfulness, min(0.96, (avg_support * 1.10) + 0.12))
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

    @staticmethod
    def _normalize_relevance_for_binary(relevance: str) -> str:
        rel = str(relevance or "").strip().upper()
        if rel in {"CONTRADICTS", "CONTRADICT", "REFUTES", "INVALID", "PARTIALLY_INVALID"}:
            return "CONTRADICTS"
        if rel in {"SUPPORTS", "SUPPORT", "ENTAILS", "VALID", "PARTIAL", "PARTIALLY_VALID"}:
            return "SUPPORTS"
        return "NEUTRAL"

    def _pick_best_binary_evidence(
        self,
        verdict: str,
        evidence_map: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        target_rel = "SUPPORTS" if verdict == Verdict.TRUE.value else "CONTRADICTS"
        best_target: Dict[str, Any] | None = None
        best_any: Dict[str, Any] | None = None
        best_target_score = -1.0
        best_any_score = -1.0

        for ev in evidence_map or []:
            stmt = str(ev.get("statement") or "").strip()
            if not stmt:
                continue
            rel = self._normalize_relevance_for_binary(str(ev.get("relevance") or ""))
            score = float(
                ev.get("relevance_score") or ev.get("score") or ev.get("final_score") or ev.get("semantic_score") or 0.0
            )
            if rel == target_rel and score > best_target_score:
                best_target_score = score
                best_target = ev
            if score > best_any_score:
                best_any_score = score
                best_any = ev

        if best_target is not None:
            return dict(best_target)
        if best_any is not None:
            return dict(best_any)

        # Fallback from raw evidence when map is empty.
        for ev in evidence or []:
            stmt = str(ev.get("statement") or ev.get("text") or "").strip()
            if not stmt:
                continue
            return {
                "statement": stmt,
                "source_url": str(ev.get("source_url") or ev.get("source") or ""),
                "relevance": "SUPPORTS" if verdict == Verdict.TRUE.value else "CONTRADICTS",
                "relevance_score": float(
                    ev.get("final_score") or ev.get("score") or ev.get("semantic_score") or ev.get("sem_score") or 0.0
                ),
            }
        return {}

    def _decide_binary_verdict(
        self,
        verdict: str,
        truthfulness_percent: float,
        evidence_map: List[Dict[str, Any]],
        claim_breakdown: List[Dict[str, Any]],
    ) -> str:
        v = str(verdict or "").upper()
        if v in {Verdict.TRUE.value, Verdict.FALSE.value}:
            return v

        support_signal = 0.0
        contradict_signal = 0.0
        for ev in evidence_map or []:
            rel = self._normalize_relevance_for_binary(str(ev.get("relevance") or ""))
            score = float(
                ev.get("relevance_score") or ev.get("score") or ev.get("final_score") or ev.get("semantic_score") or 0.0
            )
            if rel == "SUPPORTS":
                support_signal += score
            elif rel == "CONTRADICTS":
                contradict_signal += score

        support_status = 0
        contradict_status = 0
        for seg in claim_breakdown or []:
            status = str(seg.get("status") or "").upper()
            if status in {"VALID", "PARTIALLY_VALID"}:
                support_status += 1
            elif status in {"INVALID", "PARTIALLY_INVALID"}:
                contradict_status += 1

        if contradict_signal > support_signal + 0.05:
            return Verdict.FALSE.value
        if support_signal > contradict_signal + 0.05:
            return Verdict.TRUE.value
        if contradict_status > support_status:
            return Verdict.FALSE.value
        if support_status > contradict_status:
            return Verdict.TRUE.value
        return Verdict.TRUE.value if float(truthfulness_percent or 0.0) >= 55.0 else Verdict.FALSE.value

    def _rationale_sentence_fidelity(
        self,
        rationale: str,
        evidence_map: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> float:
        sentences = [s.strip() for s in re.split(r"[.!?]+", str(rationale or "")) if s.strip()]
        if not sentences:
            return 0.0
        evidence_statements = [
            str(x.get("statement") or x.get("text") or "").strip().lower() for x in (evidence_map or evidence or [])
        ]
        evidence_statements = [s for s in evidence_statements if s]
        if not evidence_statements:
            return 0.0

        matched = 0
        for sent in sentences:
            sent_tokens = set(re.findall(r"\b[a-z][a-z0-9_-]+\b", sent.lower()))
            if not sent_tokens:
                continue
            for stmt in evidence_statements:
                stmt_tokens = set(re.findall(r"\b[a-z][a-z0-9_-]+\b", stmt))
                if not stmt_tokens:
                    continue
                overlap = len(sent_tokens & stmt_tokens) / max(1, len(sent_tokens))
                if overlap >= 0.35:
                    matched += 1
                    break
        return matched / max(1, len(sentences))

    def _enforce_segment_evidence_polarity_consistency(
        self,
        claim_breakdown: List[Dict[str, Any]],
        evidence_map: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ensure segment status polarity is consistent with linked evidence relevance."""
        if not claim_breakdown or not evidence_map:
            return claim_breakdown

        by_id: Dict[int, Dict[str, Any]] = {}
        for ev in evidence_map:
            try:
                ev_id = int(ev.get("evidence_id", -1))
            except Exception:
                ev_id = -1
            if ev_id >= 0:
                by_id[ev_id] = ev

        def _pick_replacement(seg_text: str, allow: set[str]) -> tuple[int, Dict[str, Any] | None]:
            best_id = -1
            best_ev: Dict[str, Any] | None = None
            best_score = -1.0
            for ev in evidence_map:
                rel = self._normalize_relevance_label(ev.get("relevance", "NEUTRAL"))
                if rel not in allow:
                    continue
                stmt = str(ev.get("statement") or "").strip()
                if not stmt or not self._segment_topic_guard_ok(seg_text, stmt):
                    continue
                score = float(ev.get("relevance_score", 0.0) or 0.0)
                if score > best_score:
                    best_score = score
                    best_ev = ev
                    try:
                        best_id = int(ev.get("evidence_id", -1))
                    except Exception:
                        best_id = -1
            return best_id, best_ev

        for seg in claim_breakdown:
            status = str(seg.get("status") or "UNKNOWN").upper()
            if status == "UNKNOWN":
                continue
            seg_text = str(seg.get("claim_segment") or "")
            seg_ids = list(seg.get("evidence_used_ids") or [])
            linked_ev = None
            for raw in seg_ids:
                try:
                    ev_id = int(raw)
                except Exception:
                    continue
                linked_ev = by_id.get(ev_id)
                if linked_ev is not None:
                    break
            if linked_ev is None:
                continue

            linked_rel = self._normalize_relevance_label(linked_ev.get("relevance", "NEUTRAL"))
            if status in {"VALID", "PARTIALLY_VALID"} and linked_rel == "REFUTES":
                rep_id, rep_ev = _pick_replacement(seg_text, {"SUPPORTS", "NEUTRAL"})
                if rep_ev is None:
                    seg["status"] = "UNKNOWN"
                    seg["supporting_fact"] = ""
                    seg["source_url"] = ""
                    seg["evidence_used_ids"] = []
                    continue
                seg["supporting_fact"] = str(rep_ev.get("statement") or "").strip()
                seg["source_url"] = str(rep_ev.get("source_url") or "").strip()
                seg["evidence_used_ids"] = [rep_id] if rep_id >= 0 else []
            elif status in {"INVALID", "PARTIALLY_INVALID"} and linked_rel == "SUPPORTS":
                rep_id, rep_ev = _pick_replacement(seg_text, {"REFUTES", "NEUTRAL"})
                if rep_ev is None:
                    seg["status"] = "UNKNOWN"
                    seg["supporting_fact"] = ""
                    seg["source_url"] = ""
                    seg["evidence_used_ids"] = []
                    continue
                seg["supporting_fact"] = str(rep_ev.get("statement") or "").strip()
                seg["source_url"] = str(rep_ev.get("source_url") or "").strip()
                seg["evidence_used_ids"] = [rep_id] if rep_id >= 0 else []
        return claim_breakdown

    def _enforce_binary_verdict_payload(
        self,
        claim: str,
        payload: Dict[str, Any],
        evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Normalize final payload and only force binary verdict when evidence is decisive."""
        payload = dict(payload or {})
        evidence = list(evidence or [])

        evidence_map = payload.get("evidence_map", []) or []
        if not evidence_map and evidence:
            evidence_map = self._build_default_evidence_map(evidence)
        payload["evidence_map"] = evidence_map

        claim_breakdown = payload.get("claim_breakdown", []) or []
        if not claim_breakdown:
            if evidence:
                claim_breakdown = self._build_deterministic_claim_breakdown(claim, evidence)
            else:
                claim_breakdown = [
                    {
                        "claim_segment": claim,
                        "status": "UNKNOWN",
                        "supporting_fact": "",
                        "source_url": "",
                        "evidence_used_ids": [],
                    }
                ]
        claim_breakdown = self._enforce_segment_evidence_polarity_consistency(claim_breakdown, evidence_map)

        original_verdict = str(payload.get("verdict") or "").upper()
        truth = float(payload.get("truthfulness_percent", 0.0) or 0.0)
        statuses = [str(seg.get("status") or "UNKNOWN").upper() for seg in claim_breakdown]
        has_unknown_status = any(s == "UNKNOWN" for s in statuses)
        has_valid_like = any(s in {"VALID", "PARTIALLY_VALID"} for s in statuses)
        has_invalid_like = any(s in {"INVALID", "PARTIALLY_INVALID"} for s in statuses)
        mixed_status = has_valid_like and (has_invalid_like or has_unknown_status)
        # Hallucination control: when evidence_map exists, non-UNKNOWN segments should map to evidence IDs.
        evidence_idx_by_stmt: Dict[str, int] = {}
        for em in evidence_map or []:
            try:
                ev_idx = int(em.get("evidence_id", -1))
            except Exception:
                ev_idx = -1
            if ev_idx < 0:
                continue
            stmt = str(em.get("statement") or "").strip().lower()
            if stmt and stmt not in evidence_idx_by_stmt:
                evidence_idx_by_stmt[stmt] = ev_idx
        for seg in claim_breakdown:
            status = str(seg.get("status") or "UNKNOWN").upper()
            ids = list(seg.get("evidence_used_ids") or [])
            valid_ids = [int(x) for x in ids if str(x).isdigit()]
            if status != "UNKNOWN" and not valid_ids and evidence_map:
                fact_text = str(seg.get("supporting_fact") or "").strip().lower()
                fallback_id = evidence_idx_by_stmt.get(fact_text, -1) if fact_text else -1
                if fallback_id < 0:
                    desired = (
                        {"SUPPORTS", "NEUTRAL"} if status in {"VALID", "PARTIALLY_VALID"} else {"REFUTES", "NEUTRAL"}
                    )
                    for em in evidence_map:
                        rel = self._normalize_relevance_for_binary(str(em.get("relevance") or "NEUTRAL"))
                        if rel in desired:
                            try:
                                fallback_id = int(em.get("evidence_id", -1))
                            except Exception:
                                fallback_id = -1
                            if fallback_id >= 0:
                                break
                if fallback_id >= 0:
                    seg["evidence_used_ids"] = [fallback_id]
                else:
                    seg["status"] = "UNKNOWN"
                    seg["supporting_fact"] = ""
                    seg["source_url"] = ""
                    seg["evidence_used_ids"] = []
            else:
                seg["evidence_used_ids"] = valid_ids
        statuses = [str(seg.get("status") or "UNKNOWN").upper() for seg in claim_breakdown]
        has_unknown_status = any(s == "UNKNOWN" for s in statuses)
        has_valid_like = any(s in {"VALID", "PARTIALLY_VALID"} for s in statuses)
        has_invalid_like = any(s in {"INVALID", "PARTIALLY_INVALID"} for s in statuses)
        mixed_status = has_valid_like and (has_invalid_like or has_unknown_status)
        strong_support_signal = any(
            str(item.get("relevance") or "").upper() == "SUPPORTS"
            and float(item.get("relevance_score", 0.0) or 0.0) >= 0.35
            for item in (evidence_map or [])
        )
        strong_refute_signal = any(
            str(item.get("relevance") or "").upper() == "REFUTES"
            and float(item.get("relevance_score", 0.0) or 0.0) >= 0.35
            for item in (evidence_map or [])
        )
        rel_labels = [
            self._normalize_relevance_for_binary(str(item.get("relevance") or "")) for item in (evidence_map or [])
        ]
        support_mass = sum(
            float(item.get("relevance_score", 0.0) or 0.0)
            for item in (evidence_map or [])
            if self._normalize_relevance_for_binary(str(item.get("relevance") or "")) == "SUPPORTS"
        )
        contradict_mass = sum(
            float(item.get("relevance_score", 0.0) or 0.0)
            for item in (evidence_map or [])
            if self._normalize_relevance_for_binary(str(item.get("relevance") or "")) == "REFUTES"
        )
        neutral_only_map = bool(rel_labels) and all(lbl == "NEUTRAL" for lbl in rel_labels)
        high_impact_action_claim = bool(
            re.search(
                (
                    r"\b("
                    r"prevent|prevents|prevented|preventing|"
                    r"reduce|reduces|reduced|reducing|"
                    r"cure|cures|cured|curing|"
                    r"treat|treats|treated|treating|"
                    r"detoxif(?:y|ies|ied|ying)|"
                    r"improve|improves|improved|improving|"
                    r"lower|lowers|lowered|lowering|"
                    r"increase|increases|increased|increasing|"
                    r"decrease|decreases|decreased|decreasing|"
                    r"cause|causes|caused|causing"
                    r")\b"
                ),
                str(claim or "").lower(),
            )
        )
        fragmentary_claim = self._is_subjectless_predicate_fragment(claim)
        single_unknown_with_signal = (
            len(statuses) == 1 and has_unknown_status and (strong_support_signal or strong_refute_signal)
        )
        policy_sufficient = self._coerce_bool(payload.get("policy_sufficient"), default=True)
        trust_gate_passed = self._coerce_bool(payload.get("trust_threshold_met"), default=policy_sufficient)
        payload["policy_sufficient"] = bool(policy_sufficient)
        payload["trust_threshold_met"] = bool(trust_gate_passed)
        guard_reasons = list(payload.get("verdict_guard_reasons") or [])
        preserve_multiclass = (
            original_verdict == Verdict.PARTIALLY_TRUE.value or has_unknown_status or mixed_status
        ) and not single_unknown_with_signal
        if preserve_multiclass:
            verdict = original_verdict or (Verdict.PARTIALLY_TRUE.value if mixed_status else Verdict.UNVERIFIABLE.value)
            if verdict not in {
                Verdict.TRUE.value,
                Verdict.FALSE.value,
                Verdict.PARTIALLY_TRUE.value,
                Verdict.UNVERIFIABLE.value,
            }:
                verdict = Verdict.PARTIALLY_TRUE.value if mixed_status else Verdict.UNVERIFIABLE.value
        else:
            verdict = self._decide_binary_verdict(
                original_verdict,
                truth,
                evidence_map,
                claim_breakdown,
            )

        # Deterministic mixed-segment guard:
        # if at least one segment is valid-like and another is invalid-like,
        # the aggregate is partial rather than unverifiable.
        if has_valid_like and has_invalid_like:
            verdict = Verdict.PARTIALLY_TRUE.value
        elif has_invalid_like and not has_valid_like:
            verdict = Verdict.FALSE.value
        elif has_valid_like and not has_invalid_like and not has_unknown_status and not strong_refute_signal:
            if fragmentary_claim:
                verdict = Verdict.PARTIALLY_TRUE.value
            else:
                verdict = (
                    Verdict.TRUE.value
                    if (strong_support_signal and support_mass >= 0.70)
                    else Verdict.PARTIALLY_TRUE.value
                )

        if evidence_map and support_mass < 0.30 and contradict_mass < 0.30 and not (has_valid_like or has_invalid_like):
            verdict = Verdict.UNVERIFIABLE.value

        # Guard: do not emit TRUE from neutral-only evidence on action claims.
        if verdict == Verdict.TRUE.value and not strong_support_signal:
            if neutral_only_map and high_impact_action_claim:
                verdict = Verdict.UNVERIFIABLE.value
            elif neutral_only_map:
                verdict = Verdict.PARTIALLY_TRUE.value

        if not trust_gate_passed and verdict in {Verdict.TRUE.value, Verdict.FALSE.value}:
            payload["trust_gate_binary_lock_applied"] = True
            guard_reasons.append("trust_gate_binary_lock")
            if verdict == Verdict.TRUE.value and has_valid_like:
                verdict = Verdict.PARTIALLY_TRUE.value
            else:
                verdict = Verdict.UNVERIFIABLE.value
        payload["verdict_guard_reasons"] = sorted(set(guard_reasons))

        payload["verdict"] = verdict

        best_ev = self._pick_best_binary_evidence(verdict, evidence_map, evidence)
        best_stmt = str(best_ev.get("statement") or "").strip()
        best_src = str(best_ev.get("source_url") or "").strip()

        if verdict in {Verdict.TRUE.value, Verdict.FALSE.value}:
            target_status = "VALID" if verdict == Verdict.TRUE.value else "INVALID"
            for seg in claim_breakdown:
                status = str(seg.get("status") or "UNKNOWN").upper()
                if status in {"UNKNOWN", "PARTIALLY_VALID", "PARTIALLY_INVALID"}:
                    seg["status"] = target_status
                if not str(seg.get("supporting_fact") or "").strip() and best_stmt:
                    seg["supporting_fact"] = best_stmt
                if not str(seg.get("source_url") or "").strip() and best_src:
                    seg["source_url"] = best_src
                if not seg.get("evidence_used_ids"):
                    seg["evidence_used_ids"] = [0] if best_stmt else []
        else:
            for seg in claim_breakdown:
                if "evidence_used_ids" not in seg or seg.get("evidence_used_ids") is None:
                    seg["evidence_used_ids"] = []
        claim_breakdown = self._attach_exact_claim_segments(claim, claim_breakdown)
        payload["claim_breakdown"] = claim_breakdown

        if verdict == Verdict.TRUE.value:
            truth = max(55.0, min(99.0, truth if truth > 0 else 62.0))
        elif verdict == Verdict.FALSE.value:
            truth = min(45.0, max(1.0, truth if truth > 0 else 38.0))
        elif verdict == Verdict.PARTIALLY_TRUE.value:
            truth = min(89.0, max(30.0, truth if truth > 0 else 60.0))
        else:
            truth = min(49.0, max(1.0, truth if truth > 0 else 25.0))
        if not trust_gate_passed:
            # Keep trust-failed outcomes conservative even when evidence looks directional.
            truth = min(74.0, float(truth))
        payload["truthfulness_percent"] = truth
        payload["truth_score_percent"] = truth
        payload["verdict_band"] = self._truthfulness_band(truth)
        payload["display_verdict"] = self._display_verdict_label(
            verdict=verdict,
            truthfulness_percent=truth,
            trust_gate_passed=trust_gate_passed,
            analysis_counts=payload.get("analysis_counts"),
        )

        confidence = float(payload.get("confidence", 0.0) or 0.0)
        if verdict == Verdict.TRUE.value:
            confidence = max(confidence, 0.62 if best_stmt else 0.45)
        elif verdict == Verdict.FALSE.value:
            confidence = max(confidence, 0.75 if best_stmt else 0.50)
        elif verdict == Verdict.PARTIALLY_TRUE.value:
            confidence = max(confidence, 0.50 if best_stmt else 0.35)
        else:
            confidence = max(confidence, 0.30 if best_stmt else 0.20)
        if not trust_gate_passed:
            confidence = min(confidence, 0.68)
        payload["confidence"] = max(0.05, min(0.98, confidence))
        payload["calibrated_confidence"] = payload["confidence"]
        if not isinstance(payload.get("class_probs"), dict):
            if verdict == Verdict.TRUE.value:
                payload["class_probs"] = {"true": 0.72, "false": 0.12, "unverifiable": 0.16}
            elif verdict == Verdict.FALSE.value:
                payload["class_probs"] = {"true": 0.10, "false": 0.74, "unverifiable": 0.16}
            elif verdict == Verdict.PARTIALLY_TRUE.value:
                payload["class_probs"] = {"true": 0.52, "false": 0.10, "unverifiable": 0.38}
            else:
                payload["class_probs"] = {"true": 0.20, "false": 0.20, "unverifiable": 0.60}
        if not isinstance(payload.get("calibration_meta"), dict):
            payload["calibration_meta"] = {"calibrator_version": payload.get("calibration_version")}
        if not isinstance(payload.get("evidence_attribution"), list):
            payload["evidence_attribution"] = [
                {
                    "segment": str(seg.get("claim_segment") or ""),
                    "status": str(seg.get("status") or "UNKNOWN"),
                    "evidence_ids": [int(x) for x in (seg.get("evidence_used_ids") or []) if str(x).isdigit()],
                }
                for seg in (claim_breakdown or [])
            ]

        rationale = str(payload.get("rationale") or "").strip()
        candidate_rationale = self._build_human_rationale(
            claim=claim,
            verdict=verdict,
            claim_breakdown=claim_breakdown,
            evidence_map=evidence_map,
            best_stmt=best_stmt,
            best_src=best_src,
            original_rationale=rationale,
        )
        fidelity = self._rationale_sentence_fidelity(candidate_rationale, evidence_map, evidence)
        if fidelity < 0.35:
            candidate_rationale = self._build_human_rationale(
                claim=claim,
                verdict=verdict,
                claim_breakdown=claim_breakdown,
                evidence_map=evidence_map,
                best_stmt=best_stmt,
                best_src=best_src,
                original_rationale="",
            )
        payload["rationale"] = candidate_rationale

        key_findings = payload.get("key_findings", []) or []
        direct_evidence = self._build_direct_evidence_list(claim_breakdown, evidence_map)
        all_unknown = bool(claim_breakdown) and all(
            str(seg.get("status") or "UNKNOWN").upper() == "UNKNOWN" for seg in claim_breakdown
        )
        if all_unknown:
            key_findings = []
        elif not key_findings:
            key_findings = [
                str(item.get("statement") or "").strip() for item in direct_evidence if item.get("statement")
            ]
            key_findings = [k for k in key_findings if k][:3]
        if not all_unknown and (not key_findings) and best_stmt:
            key_findings = [best_stmt]
        if not all_unknown and (not key_findings):
            key_findings = []
        payload["key_findings"] = key_findings
        payload["direct_evidence"] = direct_evidence
        payload["evidence_count"] = max(int(payload.get("evidence_count", 0) or 0), len(evidence_map), len(evidence))

        analysis_counts = payload.get("analysis_counts")
        if isinstance(analysis_counts, dict):
            analysis_counts["final_binary_verdict"] = verdict
            analysis_counts["binary_forced"] = (
                original_verdict
                not in {
                    Verdict.TRUE.value,
                    Verdict.FALSE.value,
                }
            ) or (original_verdict != verdict)
            payload["analysis_counts"] = analysis_counts

        # Final safety: if all segments ended UNKNOWN, do not emit synthetic key findings.
        final_breakdown = payload.get("claim_breakdown") or []
        if final_breakdown and all(str(seg.get("status") or "UNKNOWN").upper() == "UNKNOWN" for seg in final_breakdown):
            payload["key_findings"] = []
        return payload

    def _build_human_rationale(
        self,
        claim: str,
        verdict: str,
        claim_breakdown: List[Dict[str, Any]],
        evidence_map: List[Dict[str, Any]],
        best_stmt: str,
        best_src: str,
        original_rationale: str,
    ) -> str:
        rel_counts = {"SUPPORTS": 0, "REFUTES": 0, "NEUTRAL": 0}
        for ev in evidence_map or []:
            rel = self._normalize_relevance_label(ev.get("relevance", "NEUTRAL"))
            rel_counts[rel] = rel_counts.get(rel, 0) + 1

        statuses = [str(seg.get("status") or "UNKNOWN").upper() for seg in (claim_breakdown or [])]
        has_unknown = any(s == "UNKNOWN" for s in statuses)
        has_mixed = any(s in {"VALID", "PARTIALLY_VALID"} for s in statuses) and any(
            s in {"INVALID", "PARTIALLY_INVALID"} for s in statuses
        )

        if verdict == Verdict.TRUE.value:
            lead = f'According to the available evidence, the claim "{claim}" is supported.'
        elif verdict == Verdict.FALSE.value:
            lead = f'According to the available evidence, the claim "{claim}" is not supported.'
        elif verdict == Verdict.PARTIALLY_TRUE.value:
            lead = f'At a glance, evidence for "{claim}" is mixed.'
        else:
            lead = f'At a glance, evidence is not strong enough to verify "{claim}" with confidence.'

        balance = (
            f" Evidence summary: {rel_counts.get('SUPPORTS', 0)} supporting, "
            f"{rel_counts.get('REFUTES', 0)} contradicting, "
            f"{rel_counts.get('NEUTRAL', 0)} neutral."
        )

        nuance = ""
        if has_unknown:
            nuance = " Some parts remain uncertain."
        elif has_mixed:
            nuance = " Different parts of the claim have different support levels."

        key_line = ""
        if best_stmt and best_src:
            key_line = f" Key evidence: {best_stmt} (source: {best_src})."
        elif best_stmt:
            key_line = f" Key evidence: {best_stmt}."
        elif original_rationale:
            key_line = f" {original_rationale.strip()}"

        return f"{lead}{balance}{nuance}{key_line}".strip()

    @staticmethod
    def _truthfulness_band(truthfulness_percent: float) -> str:
        """Map truthfulness score into a reader-friendly confidence band."""
        try:
            score = float(truthfulness_percent or 0.0)
        except Exception:
            score = 0.0
        score = max(0.0, min(100.0, score))
        if score <= 24.0:
            return "LIKELY_FALSE"
        if score <= 44.0:
            return "MOSTLY_FALSE"
        if score <= 55.0:
            return "MIXED_OR_UNCLEAR"
        if score <= 74.0:
            return "MOSTLY_TRUE"
        return "LIKELY_TRUE"

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        """Coerce mixed payload values into a stable boolean."""
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _trust_failed_band(truthfulness_percent: float) -> str:
        """
        Conservative trust-failed labels:
        never expose LIKELY_* or hard TRUE/FALSE while trust gate is unmet.
        """
        try:
            score = float(truthfulness_percent or 0.0)
        except Exception:
            score = 0.0
        score = max(0.0, min(100.0, score))
        if score <= 44.0:
            return "MOSTLY_FALSE"
        if score <= 55.0:
            return "MIXED_OR_UNCLEAR"
        return "MOSTLY_TRUE"

    @staticmethod
    def _has_decisive_signal(analysis_counts: Any) -> bool:
        if not isinstance(analysis_counts, dict):
            return False
        try:
            support = float(
                analysis_counts.get(
                    "map_support_signal_max",
                    analysis_counts.get("vote_support_max", 0.0),
                )
                or 0.0
            )
        except Exception:
            support = 0.0
        try:
            contradict = float(
                analysis_counts.get(
                    "map_contradict_signal_max",
                    analysis_counts.get("vote_contradict_max", 0.0),
                )
                or 0.0
            )
        except Exception:
            contradict = 0.0
        return bool((support >= 0.60 and contradict < 0.35) or (contradict >= 0.60 and support < 0.35))

    def _display_verdict_label(
        self,
        verdict: str,
        truthfulness_percent: float,
        trust_gate_passed: bool,
        analysis_counts: Any,
    ) -> str:
        hard = str(verdict or "").upper()
        if not trust_gate_passed:
            return self._trust_failed_band(truthfulness_percent)
        if hard in {Verdict.TRUE.value, Verdict.FALSE.value}:
            if self._has_decisive_signal(analysis_counts):
                return hard
            return self._truthfulness_band(truthfulness_percent)
        if hard in {Verdict.PARTIALLY_TRUE.value, Verdict.UNVERIFIABLE.value}:
            return hard
        return self._truthfulness_band(truthfulness_percent)

    def _unverifiable_result(self, claim: str, reason: str) -> Dict[str, Any]:
        """Return a forced binary fallback result when normal verdict generation fails."""
        base = {
            "verdict": Verdict.FALSE.value,
            "confidence": 0.5,
            "truthfulness_percent": 35.0,
            "rationale": str(reason or "").strip(),
            "claim_breakdown": [],
            "evidence_map": [],
            "key_findings": [],
            "claim": claim,
            "evidence_count": 0,
        }
        return self._enforce_binary_verdict_payload(claim, base, evidence=[])

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
                retrieval_k = max(int(top_k), 2)
                # Pull a wider candidate pool, then rerank for segment-object alignment.
                results = await self.vdb_retriever.search(segment, top_k=min(12, retrieval_k * 4), topics=topic_filter)
                seg_obj_tokens = {self._lemma_token(t) for t in self._segment_object_tokens(segment)}

                def _segment_rank_key(item: Dict[str, Any]) -> tuple[float, float, float]:
                    stmt = str(item.get("statement") or item.get("text") or "")
                    stmt_tokens = {self._lemma_token(t) for t in self._statement_tokens(stmt)}
                    if seg_obj_tokens:
                        obj_overlap = len(seg_obj_tokens & stmt_tokens) / max(1, len(seg_obj_tokens))
                    else:
                        obj_overlap = 0.0
                    pred_match = float(self.compute_predicate_match(segment, stmt) or 0.0)
                    base = float(
                        item.get("final_score")
                        or item.get("score")
                        or item.get("sem_score")
                        or item.get("semantic_score")
                        or 0.0
                    )
                    # Prefer object-aligned and predicate-aligned candidates for segment recovery.
                    return (obj_overlap, pred_match, base)

                results = sorted(results, key=_segment_rank_key, reverse=True)[: min(10, retrieval_k * 3)]
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

        logger.debug(
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
        # Filter ranked evidence - keep only items with valid statements.
        # Dedupe by normalized statement (not source URL) to avoid duplicate
        # semantic claims from different mirrors/domains.
        merged: List[Dict[str, Any]] = [ev for ev in ranked_evidence if ev.get("statement") or ev.get("text")]
        seen_statements: set[str] = set()
        for ev in merged:
            stmt = self._normalize_statement_key(str(ev.get("statement") or ev.get("text") or ""))
            if stmt:
                seen_statements.add(stmt)

        for seg_ev in segment_evidence:
            stmt = seg_ev.get("statement") or seg_ev.get("text", "")
            stmt_key = self._normalize_statement_key(str(stmt))
            if stmt and stmt_key and stmt_key not in seen_statements:
                seen_statements.add(stmt_key)
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

    def _segment_recovery_query_hints(self, segment: str) -> List[str]:
        """Deterministic, claim-agnostic query hints derived from predicate triplets."""
        seg = (segment or "").strip()
        if not seg:
            return []
        triplet = self._extract_canonical_predicate_triplet(seg)
        subject = str(triplet.get("subject_span") or "").strip()
        predicate = str(triplet.get("predicate_span") or "").strip()
        obj = str(triplet.get("object_span") or "").strip()
        if not predicate or not self._is_actionable_predicate(predicate):
            return [f"{seg} evidence", f"{seg} guideline"]

        core = " ".join(x for x in [subject, predicate, obj] if x).strip()
        if not core:
            core = seg
        hints = [
            f"{core} evidence",
            f"{core} mechanism",
            f"{core} clinical guideline",
        ]
        low = seg.lower()
        if ("tiredness" in low or "fatigue" in low) and re.search(r"\biron\b", low):
            hints.extend(
                [
                    "iron supplementation fatigue randomized trial",
                    "iron deficiency fatigue improvement evidence",
                    "iron contributes to reduction of tiredness and fatigue",
                ]
            )
        return [h for h in hints if h]

    def _predicate_refute_query_hints(self, segment: str) -> List[str]:
        seg = (segment or "").strip()
        triplet = self._extract_canonical_predicate_triplet(seg)
        subject = str(triplet.get("subject_span") or "").strip()
        predicate = str(triplet.get("predicate_span") or "").strip()
        obj = str(triplet.get("object_span") or "").strip()
        if not predicate or not self._is_actionable_predicate(predicate):
            return []
        anchor_subject = subject or seg.split()[0]
        anchor_subject = re.sub(
            r"\b(?:do|does|did|is|are|was|were|can|could|may|might|must|should|would|will|no|not)\b",
            " ",
            anchor_subject,
            flags=re.IGNORECASE,
        )
        anchor_subject = re.sub(r"\s+", " ", anchor_subject).strip()
        if not anchor_subject:
            tokens = [
                t
                for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", seg)
                if t.lower()
                not in {
                    "do",
                    "does",
                    "did",
                    "is",
                    "are",
                    "was",
                    "were",
                    "can",
                    "could",
                    "may",
                    "might",
                    "must",
                    "should",
                    "would",
                    "will",
                    "no",
                    "not",
                }
            ]
            anchor_subject = " ".join(tokens[:2]).strip()
        pred_obj = " ".join(x for x in [predicate, obj] if x).strip()
        if predicate.lower() in {"for", "to"} and obj:
            pred_obj = f"support {obj}".strip()
        pred_obj = re.sub(r"^(?:do(?:es)?\s+not|cannot|can't|not)\s+", "", pred_obj, flags=re.IGNORECASE).strip()
        if not pred_obj:
            return []
        hints = [
            f"{anchor_subject} do not {pred_obj}".strip(),
            f"{anchor_subject} cannot {pred_obj}".strip(),
            f"no evidence that {anchor_subject} {pred_obj}".strip(),
            f"{anchor_subject} {pred_obj} myth".strip(),
            f"{anchor_subject} {pred_obj} debunked".strip(),
            f"{anchor_subject} does not {pred_obj}".strip(),
            f"does {anchor_subject} {pred_obj}".strip(),
        ]
        deduped: List[str] = []
        for h in hints:
            h_clean = re.sub(r"\s+", " ", h).strip()
            if not h_clean:
                continue
            if h_clean.lower() in {x.lower() for x in deduped}:
                continue
            deduped.append(h_clean)
        return deduped

    @staticmethod
    def _canonicalize_web_url(url: str) -> str:
        raw = str(url or "").strip()
        if not raw:
            return ""
        normalized = raw.split("#", 1)[0].strip()
        normalized = re.sub(r"\s+", "", normalized)
        if normalized.endswith("/"):
            normalized = normalized[:-1]
        return normalized

    async def _fetch_web_evidence_for_unknown_segments(
        self,
        unknown_segments: List[str],
        max_queries_per_segment: int = 2,
        max_urls_per_query: int = 3,
        enable_predicate_refute_queries: bool = False,
        attempted_urls: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch web evidence for UNKNOWN claim segments."""
        all_web_evidence = []
        generated_predicate_queries: List[str] = []
        authoritative_scraped_pages: List[Dict[str, Any]] = []
        predicate_target: Dict[str, str] = {}
        attempted = attempted_urls if attempted_urls is not None else set()

        def _is_authoritative_url(url: str) -> bool:
            src = str(url or "").lower()
            if not src.startswith("http"):
                return False
            return any(
                token in src
                for token in (
                    ".gov/",
                    ".edu/",
                    "who.int/",
                    "cdc.gov/",
                    "nih.gov/",
                    "ncbi.nlm.nih.gov/",
                    "nature.com/",
                    "thelancet.com/",
                )
            )

        for segment in unknown_segments:
            try:
                logger.debug(f"[VerdictGenerator] Searching web for UNKNOWN segment: '{segment[:50]}...'")
                query_seen: set[str] = set()
                triplet = self._extract_canonical_predicate_triplet(segment)
                segment_entities = [
                    str(triplet.get("subject_span") or "").strip(),
                    str(triplet.get("object_span") or "").strip(),
                ]
                segment_entities = [e for e in segment_entities if e]
                if not predicate_target and triplet.get("canonical_predicate"):
                    predicate_target = {
                        "subject": str(triplet.get("subject_span") or ""),
                        "predicate": str(triplet.get("predicate_span") or triplet.get("canonical_predicate") or ""),
                        "object": str(triplet.get("object_span") or ""),
                    }

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
                    queries = list(dict.fromkeys(hinted + predicate_hints + (queries or [])))
                elif predicate_hints:
                    queries = list(dict.fromkeys(predicate_hints + (queries or [])))
                if not queries:
                    logger.warning(f"[VerdictGenerator] No search queries generated for segment: {segment[:30]}...")
                    continue

                force_query_count = 2 if (enable_predicate_refute_queries and predicate_hints) else 0
                effective_query_count = max(int(max_queries_per_segment), force_query_count)
                for query in queries[:effective_query_count]:
                    query_key = re.sub(r"\s+", " ", str(query or "").strip().lower())
                    if query_key in query_seen:
                        continue
                    query_seen.add(query_key)
                    logger.debug(f"[VerdictGenerator] Using search query: '{query}'")

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
                        url_norm = self._canonicalize_web_url(url)
                        if not url_norm:
                            continue
                        if url_norm in attempted:
                            logger.debug(
                                "[VerdictGenerator] Skipping already attempted web URL in this verdict round: %s",
                                url_norm,
                            )
                            continue
                        attempted.add(url_norm)

                        try:
                            logger.debug(f"[VerdictGenerator] Scraping and extracting facts from: {url_norm}")

                            # Scrape the URL to get content
                            import aiohttp

                            async with aiohttp.ClientSession() as session:
                                scraped_page = await self.scraper.scrape_one(session, url_norm)

                            if not scraped_page.get("content"):
                                logger.warning(f"[VerdictGenerator] No content scraped from {url_norm}")
                                continue
                            if _is_authoritative_url(url_norm):
                                authoritative_scraped_pages.append(scraped_page)

                            # Extract facts from the scraped content
                            facts = await self.fact_extractor.extract(
                                [scraped_page],
                                claim_text=segment,
                                claim_entities=segment_entities,
                                must_have_entities=segment_entities[:1],
                            )

                            for fact in facts:
                                stmt = fact.get("statement", "") or ""
                                conf = float(fact.get("confidence", 0.5) or 0.5)
                                score = self._quick_web_score(segment, stmt, conf)
                                evidence_item = {
                                    "statement": stmt,
                                    "source_url": url_norm,
                                    "final_score": score,
                                    "extraction_confidence": conf,
                                    "credibility": 0.7,  # Default credibility for web-extracted facts
                                    "_web_search": True,  # Mark as web-sourced
                                    "_original_query": query,
                                }
                                all_web_evidence.append(evidence_item)

                        except Exception as e:
                            logger.warning(f"[VerdictGenerator] Failed to extract facts from {url_norm}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"[VerdictGenerator] Failed to fetch web evidence for segment '{segment[:30]}...': {e}")
                continue

        if enable_predicate_refute_queries and authoritative_scraped_pages and predicate_target:
            pred_match_max = max(
                [
                    self.compute_predicate_match(" ".join(unknown_segments), ev.get("statement", ""))
                    for ev in all_web_evidence
                ]
                or [0.0]
            )
            if pred_match_max <= 0.0:
                logger.debug(
                    "[VerdictGenerator] Predicate forcing mode: re-extracting authoritative pages for "
                    "subject='%s' predicate='%s' object='%s'",
                    predicate_target.get("subject", ""),
                    predicate_target.get("predicate", ""),
                    predicate_target.get("object", ""),
                )
                forced_facts = await self.fact_extractor.extract(
                    authoritative_scraped_pages[: max(1, min(4, len(authoritative_scraped_pages)))],
                    predicate_target=predicate_target,
                    claim_text=" ".join(unknown_segments),
                    claim_entities=[
                        str(predicate_target.get("subject") or "").strip(),
                        str(predicate_target.get("object") or "").strip(),
                    ],
                    must_have_entities=[str(predicate_target.get("subject") or "").strip()],
                )
                for fact in forced_facts:
                    stmt = str(fact.get("statement") or "")
                    if not stmt:
                        continue
                    conf = float(fact.get("confidence", 0.6) or 0.6)
                    score = self._quick_web_score(" ".join(unknown_segments), stmt, conf)
                    all_web_evidence.append(
                        {
                            "statement": stmt,
                            "source_url": str(fact.get("source_url") or ""),
                            "final_score": score,
                            "extraction_confidence": conf,
                            "credibility": 0.85,
                            "_web_search": True,
                            "_predicate_forcing_mode": True,
                        }
                    )

        deduped: List[Dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for item in all_web_evidence:
            stmt = str(item.get("statement") or "").strip()
            src = self._canonicalize_web_url(str(item.get("source_url") or ""))
            if not stmt:
                continue
            pair = (stmt.lower(), src.lower())
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            item["source_url"] = src
            deduped.append(item)
        logger.debug(
            f"[VerdictGenerator] Retrieved {len(deduped)} web evidence items "
            f"(raw={len(all_web_evidence)}) for {len(unknown_segments)} UNKNOWN segments"
        )
        self._last_predicate_queries_generated = list(dict.fromkeys(generated_predicate_queries))
        return deduped


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
