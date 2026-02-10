"""
Trust Ranking Module: Advanced evidence ranking with component scores,
stance analysis, and post-level trust aggregation.

This module implements a comprehensive trust ranking system that evaluates evidence quality through multiple signals:
- Semantic similarity
- Source credibility
- Publication recency
- Stance/entailment analysis

It provides both per-evidence trust scores and post-level aggregation for decision-making.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import parse_qs, urlencode, urlparse

from app.core.logger import get_logger

logger = get_logger(__name__)


# Default source credibility scores (configurable)
DEFAULT_SOURCE_SCORES = {
    "who.int": 0.95,
    "cdc.gov": 0.93,
    "nih.gov": 0.90,
    "pubmed.ncbi.nlm.nih.gov": 0.92,
    "nature.com": 0.90,
    "sciencedirect.com": 0.88,
    "bbc.com": 0.75,
    "reuters.com": 0.80,
    # Add more as needed
}

# Constants for iterative trust loop
TRUST_THRESHOLD = 0.75
MIN_TRUST_IMPROVEMENT = 0.02
MAX_ITERATIONS_DEFAULT = 3
MAX_EVIDENCE_POOL = 30

# Decision constants
STOP_THRESHOLD_MET = "stop_threshold_met"
TRIGGER_CORRECTIVE = "trigger_corrective"
STOP_NO_IMPROVEMENT = "stop_no_improvement"
STOP_NO_NEW_EVIDENCE = "stop_no_new_evidence"


@dataclass
class EvidenceItem:
    """Represents a single piece of evidence for ranking."""

    statement: str
    semantic_score: float  # S_semantic: 0..1
    source_url: str
    published_at: Optional[str] = None  # ISO datetime string
    stance: str = "neutral"  # "entails", "contradicts", "neutral"
    trust: float = 0.0  # Computed trust score
    score_components: Optional[Dict[str, float]] = None  # Breakdown of trust components


@dataclass
class IterationState:
    """Tracks trust evolution per iteration for explainability."""

    iteration: int
    trust_post: float
    agreement_ratio: float
    top_sources: List[str]
    decision: str
    trust_delta: float = 0.0
    stop_reason: Optional[str] = None
    evidence_count: int = 0
    new_evidence_count: int = 0


class StanceClassifier(Protocol):
    """Protocol for stance/entailment classification."""

    def classify_stance(self, claim: str, evidence: str) -> str:
        """
        Classify the stance of evidence relative to claim.

        Returns:
            "entails", "contradicts", or "neutral"
        """
        ...


class DummyStanceClassifier:
    """Dummy implementation of stance classifier for testing/development."""

    _TARGET_STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "to",
        "for",
        "with",
        "by",
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
        "may",
        "might",
        "can",
        "could",
        "will",
        "would",
        "should",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "their",
    }

    @classmethod
    def _extract_causal_targets(cls, text: str, negated: bool) -> List[set[str]]:
        if not text:
            return []

        causal_verb_group = (
            r"(?:cause(?:s|d|ing)?|contribut(?:e|es|ed|ing)\s+to|lead(?:s|ing)?\s+to|result(?:s|ed|ing)?\s+in)"
        )

        if negated:
            pattern = re.compile(
                r"\b(?:do|does|did|can|could|may|might|must|should|would|will|is|are|was|were)?\s*"
                r"(?:not|never|no)\s+" + causal_verb_group + r"\s+([^.;:!?]+)",
                flags=re.IGNORECASE,
            )
            matches = [m.group(1) for m in pattern.finditer(text)]
        else:
            pattern = re.compile(r"\b" + causal_verb_group + r"\s+([^.;:!?]+)", flags=re.IGNORECASE)
            matches = []
            for m in pattern.finditer(text):
                prefix = text[max(0, m.start() - 20) : m.start()].lower()
                if re.search(r"\b(?:not|never|no)\s*$", prefix):
                    continue
                matches.append(m.group(1))

        targets: List[set[str]] = []
        for raw in matches:
            chunks = re.split(r"\bor\b|\band\b|,", raw, flags=re.IGNORECASE)
            for chunk in chunks:
                phrase = chunk.strip().lower()
                if not phrase:
                    continue
                if not negated and phrase.startswith(("not ", "no ", "never ")):
                    continue
                words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", phrase)
                normalized = {w for w in words if w not in cls._TARGET_STOPWORDS}
                if normalized:
                    targets.append(normalized)
        return targets

    @staticmethod
    def _targets_overlap(left: List[set[str]], right: List[set[str]]) -> bool:
        for lset in left:
            for rset in right:
                if lset & rset:
                    return True
        return False

    def classify_stance(self, claim: str, evidence: str) -> str:
        """Simple heuristic-based stance classification."""
        evidence_lower = evidence.lower()
        claim_lower = claim.lower()

        claim_neg_targets = self._extract_causal_targets(claim_lower, negated=True)
        claim_pos_targets = self._extract_causal_targets(claim_lower, negated=False)
        evidence_neg_targets = self._extract_causal_targets(evidence_lower, negated=True)
        evidence_pos_targets = self._extract_causal_targets(evidence_lower, negated=False)

        # Explicit contradiction handling for negated-vs-positive causal claims.
        if self._targets_overlap(claim_neg_targets, evidence_pos_targets) or self._targets_overlap(
            claim_pos_targets, evidence_neg_targets
        ):
            return "contradicts"
        if self._targets_overlap(claim_neg_targets, evidence_neg_targets) or self._targets_overlap(
            claim_pos_targets, evidence_pos_targets
        ):
            return "entails"

        # Fallback lexical heuristics.
        if any(word in evidence_lower for word in ["not", "false", "incorrect", "debunked"]) and any(
            word in evidence_lower for word in claim_lower.split()
        ):
            return "contradicts"
        if any(word in evidence_lower for word in ["supports", "confirms", "true", "accurate"]) and any(
            word in evidence_lower for word in claim_lower.split()
        ):
            return "entails"
        return "neutral"


class TrustRankingModule:
    """
    Advanced trust ranking module for evidence evaluation.

    Provides component-based scoring, ranking, deduplication, and post-level trust aggregation.
    """

    def __init__(
        self, stance_classifier: Optional[StanceClassifier] = None, source_scores: Optional[Dict[str, float]] = None
    ):
        self.stance_classifier = stance_classifier or DummyStanceClassifier()
        self.source_scores = source_scores or DEFAULT_SOURCE_SCORES.copy()
        # Import locally to avoid circular imports
        from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy

        self.adaptive_policy = AdaptiveTrustPolicy()

    def _normalize_url_for_dedupe(self, url: str) -> str:
        """Normalize URL for deduplication by removing fragments and tracking parameters."""
        try:
            parsed = urlparse(url)
            # Canonicalize domain: lowercase, strip www., strip port
            netloc = parsed.netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            netloc = netloc.split(":")[0]  # strip port
            # Force scheme to https
            scheme = "https"
            # Normalize path: remove trailing slash unless it's the root
            path = parsed.path.rstrip("/")
            if not path:
                path = "/"
            # Remove fragment (handled by not including it)
            # Remove common tracking parameters
            query_params = parse_qs(parsed.query)
            tracking_params = {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "fbclid",
                "gclid",
                "msclkid",
            }
            filtered_params = {k: v for k, v in query_params.items() if k not in tracking_params}
            # Sort query params deterministically
            sorted_params = dict(sorted(filtered_params.items()))
            if sorted_params:
                query = urlencode(sorted_params, doseq=True)
            else:
                query = ""
            # Reconstruct URL
            normalized = f"{scheme}://{netloc}{path}"
            if query:
                normalized += f"?{query}"
            return normalized
        except Exception:
            return url

    def compute_source_score(self, source_url: str) -> float:
        """Compute source credibility score from domain."""
        try:
            domain = urlparse(source_url).netloc.lower()
            # Strip port
            domain = domain.split(":")[0]
            # Strip www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check exact match first
            if domain in self.source_scores:
                return self.source_scores[domain]

            # Check suffix matches (e.g., sub.example.com -> example.com)
            parts = domain.split(".")
            for i in range(1, len(parts)):
                suffix = ".".join(parts[i:])
                if suffix in self.source_scores:
                    return self.source_scores[suffix]

            return 0.5
        except Exception:
            return 0.5

    def compute_recency_score(self, published_at: Optional[str]) -> float:
        """Compute recency score based on publication date."""
        if not published_at:
            return 0.5

        try:
            # Handle various ISO formats
            dt_str = published_at.replace("Z", "+00:00")
            if dt_str.endswith("+00:00"):
                dt = datetime.fromisoformat(dt_str)
            else:
                dt = datetime.fromisoformat(dt_str + "+00:00")

            now = datetime.now(dt.tzinfo)
            days_diff = (now - dt).days

            if days_diff < 180:
                return 1.0
            elif days_diff < 365:
                return 0.8
            elif days_diff < 3 * 365:
                return 0.6
            elif days_diff < 5 * 365:
                return 0.4
            else:
                return 0.3
        except Exception:
            return 0.5

    def compute_stance_score(self, stance: str) -> float:
        """Convert stance string to numerical score."""
        stance_scores = {"entails": 1.0, "neutral": 0.5, "contradicts": -1.0}
        return stance_scores.get(stance, 0.5)

    def compute_trust_evidence(self, item: EvidenceItem) -> float:
        """
        Compute trust score for a single evidence item.

        Trust = w_semantic*S_semantic + w_source*S_source + w_recency*S_recency + w_stance*stance_mapped
        where stance_mapped = (S_stance + 1) / 2  # Map -1..1 to 0..1
        """
        S_semantic = item.semantic_score
        S_source = self.compute_source_score(item.source_url)
        S_recency = self.compute_recency_score(item.published_at)
        S_stance = self.compute_stance_score(item.stance)

        # Map stance to [0,1]
        stance_mapped = (S_stance + 1) / 2

        # Weighted sum with default weights
        w_semantic = 0.30
        w_source = 0.30
        w_recency = 0.20
        w_stance = 0.20

        trust = w_semantic * S_semantic + w_source * S_source + w_recency * S_recency + w_stance * stance_mapped

        # Clamp to [0,1]
        trust = max(0.0, min(trust, 1.0))

        # Store breakdown for admin UI
        item.score_components = {
            "semantic": S_semantic,
            "source": S_source,
            "recency": S_recency,
            "stance_raw": S_stance,
            "stance_mapped": stance_mapped,
            "w_semantic": w_semantic,
            "w_source": w_source,
            "w_recency": w_recency,
            "w_stance": w_stance,
            "trust": trust,
        }

        return trust

    def rank_evidence(self, evidence_list: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Rank evidence by trust score and deduplicate.

        Returns sorted list (trust desc, then semantic desc) with duplicates removed.
        """
        # Ensure stance is set for all items
        for item in evidence_list:
            if not item.stance or item.stance not in ["entails", "contradicts", "neutral"]:
                raise ValueError(f"Stance not properly set for evidence item: {item.statement[:50]}...")

        # Compute trust for each item
        for item in evidence_list:
            item.trust = self.compute_trust_evidence(item)

        # Sort by trust desc, then semantic desc
        evidence_list.sort(key=lambda x: (-x.trust, -x.semantic_score))

        # Deduplicate by normalized URL or text hash
        seen = set()
        deduped = []

        for item in evidence_list:
            # Use normalized URL as key, fallback to statement hash
            if item.source_url:
                key = self._normalize_url_for_dedupe(item.source_url)
            else:
                key = hashlib.md5(item.statement.lower().strip().encode("utf-8")).hexdigest()

            if key not in seen:
                seen.add(key)
                deduped.append(item)

        logger.info(f"[TrustRankingModule] Ranked and deduped {len(evidence_list)} → {len(deduped)} evidence items")
        return deduped

    def compute_post_trust(self, ranked_evidence: List[EvidenceItem], top_k: int = 10) -> Dict[str, Any]:
        """
        Compute post-level trust metrics from top-k evidence.

        Args:
            ranked_evidence: List of EvidenceItem objects, assumed sorted by trust desc
            top_k: Number of top items to consider

        Returns:
            {
                "trust_post": float,  # Weighted mean trust
                "agreement_ratio": float,  # Fraction not contradicting
                "trust_grade": str,  # A+, A, B, C, D, F
                "trust_post_ci_low": float,  # Lower bound of confidence interval
                "trust_post_ci_high": float,  # Upper bound of confidence interval
                "trust_post_ci_method": str,  # "bootstrap"
                "trust_post_ci_samples": int,  # Number of bootstrap samples
                "trust_post_ci_level": float,  # Confidence level (e.g., 0.95)
                "post_breakdown": dict,  # Component breakdowns and counts
            }
        """
        evidence_items = ranked_evidence[:top_k]

        if not evidence_items:
            return {
                "trust_post": 0.0,
                "agreement_ratio": 0.0,
                "trust_grade": "F",
                "trust_post_ci_low": 0.0,
                "trust_post_ci_high": 0.0,
                "trust_post_ci_method": "bootstrap",
                "trust_post_ci_samples": 200,
                "trust_post_ci_level": 0.95,
                "post_breakdown": {
                    "semantic_mean": 0.0,
                    "source_mean": 0.0,
                    "recency_mean": 0.0,
                    "stance_mapped_mean": 0.0,
                    "evidence_used": 0,
                    "entails_count": 0,
                    "contradicts_count": 0,
                    "neutral_count": 0,
                    "top_sources": [],
                },
            }

        trusts = [item.trust for item in evidence_items]

        # Self-weighted mean: sum(trust_i * trust_i) / sum(trust_i)
        if sum(trusts) > 0:
            trust_post = sum(t * t for t in trusts) / sum(trusts)
        else:
            trust_post = 0.0

        # Agreement ratio: fraction where stance != "contradicts"
        agreement_ratio = sum(1 for item in evidence_items if item.stance != "contradicts") / len(evidence_items)

        # Assign grade
        if trust_post >= 0.85:
            grade = "A+"
        elif trust_post >= 0.75:
            grade = "A"
        elif trust_post >= 0.65:
            grade = "B"
        elif trust_post >= 0.50:
            grade = "C"
        elif trust_post >= 0.35:
            grade = "D"
        else:
            grade = "F"

        # Bootstrap confidence interval
        N = 200
        ci_level = 0.95
        ci_method = "bootstrap"
        if len(evidence_items) < 2:
            ci_low = ci_high = trust_post
        else:
            # Create deterministic seed from evidence keys
            keys = []
            for item in evidence_items:
                if item.source_url:
                    key = self._normalize_url_for_dedupe(item.source_url)
                else:
                    key = hashlib.md5(item.statement.lower().strip().encode("utf-8")).hexdigest()
                keys.append(key)
            keys.sort()  # Ensure deterministic order
            seed_str = "".join(keys)
            seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)

            bootstrap_trusts = []
            for _ in range(N):
                sample = [rng.choice(evidence_items) for _ in evidence_items]  # Sample with replacement
                trusts_sample = [item.trust for item in sample]
                if sum(trusts_sample) > 0:
                    trust_post_sample = sum(t * t for t in trusts_sample) / sum(trusts_sample)
                else:
                    trust_post_sample = 0.0
                bootstrap_trusts.append(trust_post_sample)

            bootstrap_trusts.sort()
            ci_low = bootstrap_trusts[int(0.025 * N)]
            ci_high = bootstrap_trusts[int(0.975 * N)]

        # Compute post breakdown
        if evidence_items:
            # Weighted means (weighted by trust)
            total_weight = sum(item.trust for item in evidence_items)
            if total_weight > 0:
                semantic_mean = (
                    sum(item.score_components["semantic"] * item.trust for item in evidence_items) / total_weight
                )
                source_mean = (
                    sum(item.score_components["source"] * item.trust for item in evidence_items) / total_weight
                )
                recency_mean = (
                    sum(item.score_components["recency"] * item.trust for item in evidence_items) / total_weight
                )
                stance_mapped_mean = (
                    sum(item.score_components["stance_mapped"] * item.trust for item in evidence_items) / total_weight
                )
            else:
                semantic_mean = source_mean = recency_mean = stance_mapped_mean = 0.0

            # Counts
            evidence_used = len(evidence_items)
            entails_count = sum(1 for item in evidence_items if item.stance == "entails")
            contradicts_count = sum(1 for item in evidence_items if item.stance == "contradicts")
            neutral_count = sum(1 for item in evidence_items if item.stance == "neutral")

            # Top sources (unique, sorted by frequency)
            source_counts = {}
            for item in evidence_items:
                normalized = self._normalize_url_for_dedupe(item.source_url)
                domain = normalized.split("//")[1].split("/")[0] if item.source_url else "unknown"
                source_counts[domain] = source_counts.get(domain, 0) + 1
            top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]  # Top 5
            top_sources = [s[0] for s in top_sources]

            post_breakdown = {
                "semantic_mean": semantic_mean,
                "source_mean": source_mean,
                "recency_mean": recency_mean,
                "stance_mapped_mean": stance_mapped_mean,
                "evidence_used": evidence_used,
                "entails_count": entails_count,
                "contradicts_count": contradicts_count,
                "neutral_count": neutral_count,
                "top_sources": top_sources,
            }
        else:
            post_breakdown = {
                "semantic_mean": 0.0,
                "source_mean": 0.0,
                "recency_mean": 0.0,
                "stance_mapped_mean": 0.0,
                "evidence_used": 0,
                "entails_count": 0,
                "contradicts_count": 0,
                "neutral_count": 0,
                "top_sources": [],
            }

        return {
            "trust_post": trust_post,
            "agreement_ratio": agreement_ratio,
            "trust_grade": grade,
            "trust_post_ci_low": ci_low,
            "trust_post_ci_high": ci_high,
            "trust_post_ci_method": ci_method,
            "trust_post_ci_samples": N,
            "trust_post_ci_level": ci_level,
            "post_breakdown": post_breakdown,
        }

    def decide(self, evidence_list: List[EvidenceItem], threshold: float = 0.75) -> str:
        """
        Make decision based on post-level trust.

        Returns:
            "proceed_to_generation" if trust_post >= threshold
            "trigger_corrective_retrieval" otherwise
        """
        ranked = self.rank_evidence(evidence_list)
        post_trust = self.compute_post_trust(ranked)

        if post_trust["trust_post"] >= threshold:
            decision = "proceed_to_generation"
        else:
            decision = "trigger_corrective_retrieval"

        logger.info(
            f"[TrustRankingModule] Decision: {decision} "
            f"(trust_post={post_trust['trust_post']:.3f}, threshold={threshold})"
        )
        return decision

    def compute_adaptive_post_trust(
        self, claim: str, ranked_evidence: List[EvidenceItem], top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Compute adaptive post-level trust for multi-part claims.

        Uses the adaptive trust policy to handle complex claims with dynamic thresholds.

        Args:
            claim: The full claim text
            ranked_evidence: List of EvidenceItem objects, assumed sorted by trust desc
            top_k: Number of top items to consider

        Returns:
            Dictionary with adaptive trust metrics and decision
        """
        return self.adaptive_policy.compute_adaptive_trust(claim, ranked_evidence, top_k)

    def decide_adaptive(self, claim: str, evidence_list: List[EvidenceItem], threshold: float = 0.75) -> str:
        """
        Make adaptive decision based on claim complexity and evidence sufficiency.

        Uses adaptive trust policy for multi-part claims.

        Returns:
            "proceed_to_generation" if evidence is sufficient
            "trigger_corrective_retrieval" otherwise
        """
        ranked = self.rank_evidence(evidence_list)
        adaptive_trust = self.compute_adaptive_post_trust(claim, ranked)

        if adaptive_trust["is_sufficient"]:
            decision = "proceed_to_generation"
        else:
            decision = "trigger_corrective_retrieval"

        logger.info(
            f"[TrustRankingModule] Adaptive decision: {decision} "
            f"(sufficient={adaptive_trust['is_sufficient']}, trust_post={adaptive_trust['trust_post']:.3f})"
        )
        return decision

    def classify_stance_for_evidence(self, claim: str, evidence_list: List[EvidenceItem]) -> None:
        """
        Classify stance for evidence items that don't have it set.

        Modifies evidence_list in-place.
        """
        for item in evidence_list:
            if not item.stance or item.stance == "neutral":  # Only classify if not already set or neutral
                item.stance = self.stance_classifier.classify_stance(claim, item.statement)


# Backward compatibility: Keep the old TrustRanker class
class TrustRanker:
    """
    Legacy class for grade assignment. Use TrustRankingModule for new functionality.
    """

    GRADES = ["A+", "A", "B", "C", "D", "F"]

    @staticmethod
    def assign_grade(final_score: float) -> str:
        """Map final_score [0.0, 1.0] to a trust grade."""
        final_score = max(0.0, min(final_score, 1.0))

        if final_score >= 0.90:
            return "A+"
        elif final_score >= 0.80:
            return "A"
        elif final_score >= 0.70:
            return "B"
        elif final_score >= 0.60:
            return "C"
        elif final_score >= 0.50:
            return "D"
        else:
            return "F"

    @staticmethod
    def assign_semantic_confidence(semantic_score: float) -> str:
        """Assess semantic similarity confidence level."""
        semantic_score = max(0.0, min(semantic_score, 1.0))

        if semantic_score >= 0.9:
            return "HIGH"
        elif semantic_score >= 0.75:
            return "GOOD"
        elif semantic_score >= 0.6:
            return "FAIR"
        elif semantic_score >= 0.4:
            return "LOW"
        else:
            return "NONE"

    @staticmethod
    def enrich_ranked_results(ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add trust grades and semantic confidence to ranked results."""
        enriched = []

        for result in ranked_results:
            final_score = result.get("final_score", 0.0)
            sem_score = result.get("sem_score", 0.0)
            credibility = result.get("credibility", 0.5)
            entity_overlap = result.get("entity_overlap", 0.0)
            recency = result.get("recency", 0.0)

            grade = TrustRanker.assign_grade(final_score)
            sem_confidence = TrustRanker.assign_semantic_confidence(sem_score)

            # Build rationale
            factors = []
            if sem_score >= 0.6:
                factors.append(f"high semantic match ({sem_score:.2f})")
            elif sem_score >= 0.4:
                factors.append(f"fair semantic match ({sem_score:.2f})")

            if credibility >= 0.75:
                factors.append(f"trusted source ({credibility:.2f})")
            elif credibility >= 0.5:
                factors.append(f"moderate credibility ({credibility:.2f})")

            if entity_overlap >= 0.7:
                factors.append(f"strong entity overlap ({entity_overlap:.2f})")
            elif entity_overlap >= 0.3:
                factors.append(f"some entity match ({entity_overlap:.2f})")

            if recency >= 0.5:
                factors.append(f"recent source ({recency:.2f})")

            rationale = "; ".join(factors) if factors else "low confidence across all signals"

            enriched.append(
                {
                    **result,
                    "grade": grade,
                    "semantic_confidence": sem_confidence,
                    "grade_rationale": rationale,
                }
            )

        logger.info(f"[TrustRanker] Enriched {len(enriched)} results with grades.")
        return enriched

    @staticmethod
    def filter_by_grade(enriched_results: List[Dict[str, Any]], min_grade: str = "C") -> List[Dict[str, Any]]:
        """Filter results by minimum grade."""
        grade_order = ["A+", "A", "B", "C", "D", "F"]
        if min_grade not in grade_order:
            min_grade = "C"

        min_index = grade_order.index(min_grade)
        filtered = [r for r in enriched_results if grade_order.index(r.get("grade", "F")) <= min_index]

        logger.info(f"[TrustRanker] Filtered {len(enriched_results)} → {len(filtered)} results (min_grade={min_grade})")
        return filtered

    @staticmethod
    def _grade_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count distribution of grades in results."""
        dist = {}
        for result in results:
            grade = result.get("grade")
            if grade:
                dist[grade] = dist.get(grade, 0) + 1
        return dist
