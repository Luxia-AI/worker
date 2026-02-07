"""
Adaptive Trust Policy Module: Handles multi-part claims with dynamic trust thresholds.

This module implements an adaptive trust evaluation system that:
- Decomposes complex claims into subclaims
- Calculates coverage, diversity, and agreement metrics
- Applies dynamic gating rules based on claim complexity
- Provides evidence insufficiency detection
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.ranking.subclaim_coverage import compute_subclaim_coverage
from app.services.ranking.trust_ranker import EvidenceItem

logger = get_logger(__name__)


class AdaptiveTrustPolicy:
    """
    Adaptive trust policy for multi-part claims.

    Handles claim decomposition, metric calculation, and dynamic threshold application.
    """

    def __init__(self):
        # Gating rule thresholds
        self.COVERAGE_THRESHOLD_HIGH = float(os.getenv("ADAPTIVE_COVERAGE_THRESHOLD_HIGH", "0.60"))
        self.COVERAGE_THRESHOLD_MEDIUM = float(os.getenv("ADAPTIVE_COVERAGE_THRESHOLD_MEDIUM", "0.40"))
        self.DIVERSITY_THRESHOLD_HIGH = float(os.getenv("ADAPTIVE_DIVERSITY_THRESHOLD_HIGH", "0.70"))
        self.AGREEMENT_THRESHOLD_HIGH = float(os.getenv("ADAPTIVE_AGREEMENT_THRESHOLD_HIGH", "0.80"))
        self.MIN_EVIDENCE_COUNT = int(os.getenv("ADAPTIVE_MIN_EVIDENCE_COUNT", "3"))
        self.HIGH_DIVERSITY_FOR_LOW_COUNT = float(os.getenv("ADAPTIVE_HIGH_DIVERSITY_FOR_LOW_COUNT", "0.85"))
        self.HIGH_RELEVANCE_FOR_LOW_COUNT = float(os.getenv("ADAPTIVE_HIGH_RELEVANCE_FOR_LOW_COUNT", "0.75"))

        # Scoring weights
        self.COVERAGE_WEIGHT = 0.5
        self.BASE_WEIGHT = 0.5

    def decompose_claim(self, claim: str) -> List[str]:
        """
        Decompose a complex claim into subclaims.

        Uses linguistic patterns to identify separable components:
        - Multiple sentences
        - Conjunctive phrases (and, or, but)
        - List-like structures
        - Comparative statements

        Args:
            claim: The full claim text

        Returns:
            List of subclaim strings
        """
        if not claim or not claim.strip():
            return []

        claim = claim.strip()

        # Split on sentence boundaries first
        sentences = re.split(r"(?<=[.!?])\s+", claim)

        subclaims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check for conjunctive decomposition
            conjunctives = self._split_on_conjunctives(sentence)
            if conjunctives:
                subclaims.extend(conjunctives)
            else:
                # Check for list-like structures
                lists = self._split_on_lists(sentence)
                if lists:
                    subclaims.extend(lists)
                else:
                    # Check for comparative structures
                    comparatives = self._split_on_comparatives(sentence)
                    if comparatives:
                        subclaims.extend(comparatives)
                    else:
                        # Single atomic claim
                        subclaims.append(sentence)

        # Remove duplicates while preserving order
        seen = set()
        unique_subclaims = []
        for subclaim in subclaims:
            normalized = subclaim.lower().strip()
            if normalized not in seen and len(normalized) > 10:  # Minimum length filter
                seen.add(normalized)
                unique_subclaims.append(subclaim)

        logger.info(f"Decomposed claim into {len(unique_subclaims)} subclaims: {unique_subclaims}")
        return unique_subclaims

    def _split_on_conjunctives(self, text: str) -> List[str]:
        """Split on conjunctive words while preserving context."""

        def _clean(t: str) -> str:
            return re.sub(r"\s+", " ", t).strip(" ,.")

        # Enumeration-aware expansion for list claims with a shared predicate.
        pred = re.search(
            (
                r"\b(helps?|prevents?|reduces?|increases?|causes?|improves?|worsens?|protects?|"
                r"associated with|linked to|leads to)\b"
            ),
            text,
            flags=re.IGNORECASE,
        )
        if pred:
            head = _clean(text[: pred.start()])
            tail = _clean(text[pred.start() :])
            # Only treat as enumeration when list punctuation exists.
            if "," in head:
                normalized = re.sub(r"\s*,\s*and\s+", ", ", head, flags=re.IGNORECASE)
                normalized = re.sub(r"\s+and\s+", ", ", normalized, flags=re.IGNORECASE)
                items = [_clean(x) for x in normalized.split(",") if _clean(x)]
                if len(items) >= 2:
                    first = items[0]
                    q = re.search(r"\b(rich in|low in|high in|with|without|deficient in)\b", first, flags=re.IGNORECASE)
                    subject_root = _clean(first[: q.start()]) if q else _clean(" ".join(first.split()[:2]))
                    qualifier_prefix = (q.group(1).strip() + " ") if q else ""
                    out: List[str] = []
                    for idx, item in enumerate(items):
                        phrase = item
                        if (
                            idx > 0
                            and subject_root
                            and re.match(
                                r"^(rich in|low in|high in|with|without|deficient in)\b", item, flags=re.IGNORECASE
                            )
                        ):
                            phrase = f"{subject_root} {item}"
                        elif idx > 0 and len(item.split()) <= 4:
                            if qualifier_prefix and subject_root:
                                phrase = f"{subject_root} {qualifier_prefix}{item}"
                            elif subject_root:
                                phrase = f"{subject_root} {item}"
                            elif qualifier_prefix:
                                phrase = qualifier_prefix + item
                        out.append(_clean(f"{phrase} {tail}"))
                    return out

        conjunctives = [" and ", " or ", " but ", " however ", " although ", " while "]

        for conj in conjunctives:
            if conj in text.lower():
                parts = [_clean(p) for p in text.split(conj) if _clean(p)]
                if len(parts) > 1:
                    # Reconstruct with conjunction for context
                    result = []
                    for i, part in enumerate(parts):
                        if i == 0:
                            result.append(part)
                        else:
                            result.append(f"{conj.strip()} {part}")
                    return result
        return []

    def _split_on_lists(self, text: str) -> List[str]:
        """Split on list-like structures (numbered, bulleted, etc.)."""
        # Numbered lists: 1., 2., (1), (2), etc.
        numbered = re.split(r"\s+(?:\d+\.|\(\d+\))\s+", text)
        if len(numbered) > 1:
            return [item.strip() for item in numbered if item.strip()]

        # Bulleted lists: -, *, •
        bulleted = re.split(r"\s*[-*•]\s+", text)
        if len(bulleted) > 1:
            return [item.strip() for item in bulleted if item.strip()]

        return []

    def _split_on_comparatives(self, text: str) -> List[str]:
        """Split on comparative structures (higher than, more than, etc.)."""
        # Pattern: "more/less X than Y"
        m = re.search(
            r"\b(more|less|higher|lower|greater|fewer|better|worse)\b(.+?)\bthan\b(.+)$",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            left = text[: m.start()].strip()
            comp_mid = (m.group(1) + m.group(2)).strip()
            right = m.group(3).strip()
            if left and comp_mid and right:
                return [left, f"{comp_mid} than {right}".strip()]

        comparatives = [
            " more than ",
            " less than ",
            " higher than ",
            " lower than ",
            " greater than ",
            " fewer than ",
            " better than ",
            " worse than ",
        ]

        for comp in comparatives:
            if comp in text.lower():
                parts = text.split(comp)
                if len(parts) == 2:
                    return [parts[0].strip(), f"{comp.strip()} {parts[1].strip()}"]
        return []

    def calculate_coverage(self, subclaims: List[str], evidence_list: List[EvidenceItem]) -> float:
        """
        Calculate coverage: fraction of subclaims with supporting evidence.

        Args:
            subclaims: List of subclaim strings
            evidence_list: List of evidence items

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if not subclaims:
            return 0.0

        cov = compute_subclaim_coverage(subclaims, evidence_list, partial_weight=0.5)
        coverage = float(cov.get("coverage", 0.0))
        weighted = float(cov.get("weighted_covered", 0.0))
        logger.info("Coverage: weighted=%.2f/%d = %.2f", weighted, len(subclaims), coverage)
        for d in cov.get("details", []):
            logger.info(
                "[AdaptiveTrustPolicy][Coverage] subclaim=%d best_evidence_id=%s "
                "relevance=%.3f overlap=%.3f anchors=%d/%d status=%s",
                d.get("subclaim_id"),
                d.get("best_evidence_id"),
                float(d.get("relevance_score", 0.0)),
                float(d.get("overlap", 0.0)),
                int(d.get("anchors_matched", 0)),
                int(d.get("anchors_required", 0)),
                d.get("status"),
            )
        return coverage

    def _evidence_covers_subclaim(self, subclaim: str, evidence: EvidenceItem) -> bool:
        """Stricter coverage: require meaningful overlap AND strong semantic match."""
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
        sub_words = [w for w in re.findall(r"\b\w+\b", subclaim.lower()) if w not in stop]
        ev_words = [w for w in re.findall(r"\b\w+\b", evidence.statement.lower()) if w not in stop]
        if not sub_words or not ev_words:
            return False

        sub_set, ev_set = set(sub_words), set(ev_words)
        overlap = len(sub_set & ev_set)
        # Require at least 2 overlapping "content" words to avoid accidental matches
        if overlap < 2:
            return False

        overlap_ratio = overlap / max(1, len(sub_set))
        sem = max(getattr(evidence, "semantic_score", 0.0), getattr(evidence, "trust", 0.0))
        # Require both: decent overlap and strong semantic/trust match
        return overlap_ratio >= 0.20 and sem >= 0.45

    def calculate_diversity(self, evidence_list: List[EvidenceItem]) -> float:
        """
        Calculate evidence diversity based on source variety.

        Args:
            evidence_list: List of evidence items

        Returns:
            Diversity score (0.0 to 1.0)
        """
        if not evidence_list:
            return 0.0

        # Extract unique domains
        domains = set()
        for evidence in evidence_list:
            try:
                from urllib.parse import urlparse

                domain = urlparse(evidence.source_url).netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
                domains.add(domain)
            except Exception:
                domains.add("unknown")

        # Diversity = unique domains / total evidence (with diminishing returns)
        unique_domains = len(domains)
        total_evidence = len(evidence_list)

        if total_evidence <= 1:
            return 0.0

        # Use entropy-like formula: diversity = 1 - (1/unique_domains) * log(total_evidence/unique_domains)
        import math

        if unique_domains >= total_evidence:
            diversity = 1.0
        else:
            ratio = total_evidence / unique_domains
            diversity = 1.0 - (1.0 / unique_domains) * math.log(ratio)

        diversity = max(0.0, min(1.0, diversity))
        logger.info(f"Diversity: {unique_domains} unique domains from {total_evidence} evidence = {diversity:.2f}")
        return diversity

    def calculate_agreement(self, evidence_list: List[EvidenceItem]) -> float:
        """
        Calculate agreement ratio: fraction of evidence that doesn't contradict.

        Args:
            evidence_list: List of evidence items

        Returns:
            Agreement ratio (0.0 to 1.0)
        """
        if not evidence_list:
            return 0.0

        non_contradicting = sum(1 for e in evidence_list if getattr(e, "stance", "unknown") != "contradicts")
        agreement = non_contradicting / len(evidence_list)

        logger.info(f"Agreement: {non_contradicting}/{len(evidence_list)} = {agreement:.2f}")
        return agreement

    def apply_gating_rules(
        self,
        coverage: float,
        diversity: float,
        agreement: float,
        evidence_count: int | None = None,
        avg_relevance: float | None = None,
    ) -> bool:
        """
        Apply adaptive gating rules to determine if evidence is sufficient.

        Rules:
        1. High coverage (>= 0.6) -> sufficient
        2. Medium coverage (>= 0.4) AND high diversity (>= 0.7) -> sufficient
        3. Otherwise -> insufficient

        Args:
            coverage: Coverage ratio
            diversity: Diversity score
            agreement: Agreement ratio

        Returns:
            True if evidence is sufficient, False otherwise
        """
        if evidence_count is not None and evidence_count < self.MIN_EVIDENCE_COUNT:
            if (
                avg_relevance is not None
                and diversity >= self.HIGH_DIVERSITY_FOR_LOW_COUNT
                and avg_relevance >= self.HIGH_RELEVANCE_FOR_LOW_COUNT
            ):
                logger.info(
                    "Gating: LOW-COUNT override PASS (count=%d, diversity=%.2f, avg_relevance=%.2f)",
                    evidence_count,
                    diversity,
                    avg_relevance,
                )
            else:
                logger.info(
                    "Gating: FAIL - insufficient evidence count (%d < %d)",
                    evidence_count,
                    self.MIN_EVIDENCE_COUNT,
                )
                return False

        if coverage >= self.COVERAGE_THRESHOLD_HIGH and agreement >= self.AGREEMENT_THRESHOLD_HIGH:
            logger.info(
                f"Gating: PASS - High coverage ({coverage:.2f} >= {self.COVERAGE_THRESHOLD_HIGH}) "
                f"+ high agreement ({agreement:.2f} >= {self.AGREEMENT_THRESHOLD_HIGH})"
            )
            return True

        if (
            coverage >= self.COVERAGE_THRESHOLD_MEDIUM
            and diversity >= self.DIVERSITY_THRESHOLD_HIGH
            and agreement >= self.AGREEMENT_THRESHOLD_HIGH
        ):
            logger.info(
                f"Gating: PASS - Medium coverage ({coverage:.2f}) + high diversity ({diversity:.2f}) "
                f"+ high agreement ({agreement:.2f})"
            )
            return True

        logger.info(f"Gating: FAIL - coverage={coverage:.2f}, diversity={diversity:.2f}, agreement={agreement:.2f}")
        return False

    def compute_adaptive_trust(self, claim: str, evidence_list: List[EvidenceItem], top_k: int = 10) -> Dict[str, Any]:
        """
        Compute adaptive trust score for multi-part claims.

        Args:
            claim: The full claim text
            evidence_list: List of evidence items
            top_k: Number of top evidence items to consider

        Returns:
            Dictionary with trust metrics and decision
        """
        # Decompose claim into subclaims
        subclaims = self.decompose_claim(claim)
        num_subclaims = len(subclaims)

        # Use top-k evidence
        top_evidence = evidence_list[:top_k] if evidence_list else []

        # Calculate metrics
        coverage = self.calculate_coverage(subclaims, top_evidence)
        diversity = self.calculate_diversity(top_evidence)
        agreement = self.calculate_agreement(top_evidence)
        avg_relevance = (
            sum(
                max(float(getattr(e, "semantic_score", 0.0) or 0.0), float(getattr(e, "trust", 0.0) or 0.0))
                for e in top_evidence
            )
            / max(1, len(top_evidence))
            if top_evidence
            else 0.0
        )

        # Apply gating rules
        is_sufficient = self.apply_gating_rules(
            coverage,
            diversity,
            agreement,
            evidence_count=len(top_evidence),
            avg_relevance=avg_relevance,
        )

        # Compute adaptive trust score
        if not top_evidence:
            trust_post = 0.0
        else:
            # Get individual evidence trust scores
            subclaim_trusts = []
            cov = compute_subclaim_coverage(subclaims, top_evidence, partial_weight=0.5)
            for d in cov.get("details", []):
                best_idx = int(d.get("best_evidence_id", -1))
                if best_idx < 0 or best_idx >= len(top_evidence):
                    continue
                weight = float(d.get("weight", 0.0))
                if weight <= 0.0:
                    continue
                try:
                    best_trust = float(getattr(top_evidence[best_idx], "trust", 0.0) or 0.0)
                except Exception:
                    best_trust = 0.0
                if best_trust > 0.0:
                    subclaim_trusts.append(best_trust * weight)

            if subclaim_trusts:
                mean_subclaim_trust = sum(subclaim_trusts) / len(subclaim_trusts)
                # Adaptive formula: trust_post = mean(subclaim_trust) * (0.5 + 0.5*coverage)
                trust_post = mean_subclaim_trust * (self.BASE_WEIGHT + self.COVERAGE_WEIGHT * coverage)
            else:
                trust_post = 0.0

        # Determine verdict state
        if not is_sufficient:
            verdict_state = "evidence_insufficiency"
        elif trust_post >= 0.8:
            verdict_state = "confirmed"
        elif trust_post >= 0.6:
            verdict_state = "provisional"
        else:
            verdict_state = "revoked"

        result = {
            "trust_post": trust_post,
            "coverage": coverage,
            "diversity": diversity,
            "agreement": agreement,
            "num_subclaims": num_subclaims,
            "subclaims": subclaims,
            "evidence_used": len(top_evidence),
            "is_sufficient": is_sufficient,
            "verdict_state": verdict_state,
            "adaptive_metrics": {
                "coverage_threshold_high": self.COVERAGE_THRESHOLD_HIGH,
                "coverage_threshold_medium": self.COVERAGE_THRESHOLD_MEDIUM,
                "diversity_threshold_high": self.DIVERSITY_THRESHOLD_HIGH,
                "agreement_threshold_high": self.AGREEMENT_THRESHOLD_HIGH,
                "min_evidence_count": self.MIN_EVIDENCE_COUNT,
                "low_count_diversity_override": self.HIGH_DIVERSITY_FOR_LOW_COUNT,
                "low_count_relevance_override": self.HIGH_RELEVANCE_FOR_LOW_COUNT,
            },
        }

        logger.info(
            f"Adaptive trust result: trust_post={trust_post:.3f}, "
            f"coverage={coverage:.2f}, sufficient={is_sufficient}"
        )

        # Hard relevance floor: don't skip web if evidence is weak overall
        sem_scores = [getattr(e, "semantic_score", 0.0) for e in top_evidence] if top_evidence else [0.0]
        trust_scores = [getattr(e, "trust", 0.0) for e in top_evidence] if top_evidence else [0.0]
        top_sem = max(sem_scores) if sem_scores else 0.0
        top_trust = max(trust_scores) if trust_scores else 0.0
        # Performance optimization: avoid overriding when coverage+agreement are strong.
        strong_coverage = coverage >= 0.95
        strong_agreement = agreement >= 0.9
        if (
            is_sufficient
            and not (strong_coverage and strong_agreement)
            and (trust_post < 0.55 or top_sem < 0.60 or top_trust < 0.55)
        ):
            logger.info(
                "[AdaptiveTrustPolicy] Overriding sufficient=False due to weak relevance "
                f"(trust_post={trust_post:.3f}, top_sem={top_sem:.3f}, top_trust={top_trust:.3f})"
            )
            result["is_sufficient"] = False
            result["verdict_state"] = "evidence_insufficiency"

        return result
