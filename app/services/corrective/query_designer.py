"""
Corrective Query Designer

Production-oriented query planning module for health misinformation claims.
It provides:
- Rule-based claim typing
- Lightweight entity extraction
- Query template generation (6-8 weighted queries)
- Drift-driven negative learning (counted + threshold)
- Efficient quality gating and adaptive top-N selection

Integration hooks:
- build_plan(claim: str) -> QueryPlan
- register_drift_from_url(claim_type: str, url: str, title: str, snippet: str) -> None
"""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from urllib.parse import urlparse

from app.config.trusted_domains import TRUSTED_ROOT_DOMAINS, is_trusted_domain
from app.core.logger import get_logger

logger = get_logger(__name__)

CLAIM_TYPES = {
    "EFFICACY",
    "SAFETY",
    "STATISTICAL",
    "NUMERIC_COMPARISON",
    "MECHANISM",
    "MYTH",
    "UNIQUENESS",
    "GENERAL",
}


@dataclass(slots=True)
class QuerySpec:
    q: str
    goal: str
    weight: float


@dataclass(slots=True)
class ClaimEntities:
    topic: List[str] = field(default_factory=list)
    outcome: List[str] = field(default_factory=list)
    population: List[str] = field(default_factory=list)
    comparator: List[str] = field(default_factory=list)
    anchors: List[str] = field(default_factory=list)
    synonyms: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "topic": self.topic,
            "outcome": self.outcome,
            "population": self.population,
            "comparator": self.comparator,
            "anchors": self.anchors,
            "synonyms": self.synonyms,
        }


@dataclass(slots=True)
class QueryPlan:
    claim: str
    claim_type: str
    extracted_entities: ClaimEntities
    queries: List[QuerySpec]
    negatives: List[str]
    strategy_notes: List[str]

    def to_log_dict(self) -> Dict[str, object]:
        return {
            "claim": self.claim,
            "claim_type": self.claim_type,
            "extracted_entities": self.extracted_entities.to_dict(),
            "queries": [asdict(q) for q in self.queries],
            "negatives": self.negatives,
            "strategy_notes": self.strategy_notes,
        }


@dataclass(slots=True)
class QualityGateResult:
    passed: bool
    score: float
    reasons: List[str] = field(default_factory=list)


class CorrectiveQueryDesigner:
    # ----------------------------
    # Stopwords / keywords
    # ----------------------------
    _STOPWORDS = {
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
        "these",
        "those",
        "has",
        "have",
        "had",
        "does",
        "do",
        "did",
        "as",
        "it",
        "its",
        "their",
        "there",
        "into",
        "from",
        "about",
        "up",
        "down",
        "than",
    }

    _CLAIM_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "EFFICACY": (
            "effective",
            "efficacy",
            "works",
            "work",
            "prevent",
            "prevents",
            "reduce",
            "reduces",
            "improve",
            "improves",
            "benefit",
            "treat",
            "treats",
            "cure",
            "cures",
            "reverses",
        ),
        "SAFETY": (
            "safe",
            "safety",
            "harm",
            "harmful",
            "toxic",
            "risk",
            "adverse",
            "side effect",
            "side effects",
            "dangerous",
            "injury",
            "fatal",
            "infertility",
        ),
        "STATISTICAL": (
            "rate",
            "rates",
            "incidence",
            "prevalence",
            "odds",
            "percent",
            "percentage",
            "probability",
            "risk ratio",
            "hazard ratio",
            "proportion",
            "statistics",
        ),
        "NUMERIC_COMPARISON": (
            "more than",
            "less than",
            "fewer than",
            "greater than",
            "higher than",
            "lower than",
            "compared to",
            "versus",
            "vs",
            "number of",
            "count",
            "population",
            "estimate",
        ),
        "MECHANISM": (
            "mechanism",
            "pathway",
            "via",
            "through",
            "because",
            "trigger",
            "mediated",
            "binds",
            "metabolism",
            "immune",
        ),
        "MYTH": (
            "myth",
            "hoax",
            "fake",
            "false",
            "debunk",
            "conspiracy",
            "rumor",
            "misinformation",
            "not true",
        ),
        "UNIQUENESS": (
            "unique",
            "uniqueness",
            "fingerprint",
            "tongue print",
            "distinct",
            "individuality",
            "no two",
            "one of a kind",
            "identical",
            "biometric",
            "identification",
        ),
    }

    _POPULATION_TERMS = {
        "adults",
        "adult",
        "children",
        "child",
        "babies",
        "baby",
        "infants",
        "infant",
        "newborns",
        "newborn",
        "pregnant",
        "women",
        "men",
        "elderly",
        "older adults",
        "patients",
        "humans",
        "people",
        "teens",
        "adolescents",
        "everyone",
    }

    # ----------------------------
    # Negatives: keep safe defaults small; rely on drift for the rest
    # ----------------------------
    _NEGATIVE_BASE = {
        "reddit",
        "quora",
        "facebook",
        "pinterest",
        "youtube",
    }

    _NEGATIVE_BY_TYPE = {
        "EFFICACY": {"testimonial", "opinion"},
        "SAFETY": {"promo", "advertisement"},
        "STATISTICAL": {"lottery"},
        "NUMERIC_COMPARISON": {"lottery", "gambling"},
        "MECHANISM": {"astrology", "spiritual"},
        "MYTH": {"satire"},
        "UNIQUENESS": {"username", "logo", "tattoo", "wallpaper"},
        "GENERAL": set(),
    }

    # ----------------------------
    # Trusted-domain routing derived from canonical trusted roots.
    # ----------------------------
    _TRUSTED_ROUTE_HINTS_BY_TYPE: Dict[str, Tuple[str, ...]] = {
        "EFFICACY": ("pubmed", "cochrane", "clinicaltrials", "cdc", "nih", "who", "jamanetwork", "bmj", "nejm"),
        "SAFETY": ("fda", "cdc", "nih", "who", "ema", "ecdc", "nhs", "clinicaltrials"),
        "STATISTICAL": ("ourworldindata", "ihme", "healthdata", "worldbank", "cdc", "who"),
        "NUMERIC_COMPARISON": ("pubmed", "pmc", "ncbi", "plos", "nature", "science", "ourworldindata", "worldbank"),
        "MECHANISM": ("pubmed", "pmc", "nih", "nlm", "nature", "science", "plos"),
        "MYTH": ("cdc", "who", "nih", "nhs", "nice"),
        "UNIQUENESS": ("pubmed", "pmc", "nih", "nlm", "nature", "plos"),
        "GENERAL": ("pubmed", "cdc", "nih", "who", "nhs"),
    }

    # Low-signal patterns: used for scoring and drift
    _LOW_SIGNAL_PATTERNS = (
        "manual",
        "policy",
        "handbook",
        "template",
        "brochure",
        "procurement",
    )

    # Explicit synonym seed map (tiny, expand later)
    _ENTITY_SYNONYMS: Dict[str, Tuple[str, ...]] = {
        "vaccine": ("vaccination", "immunization", "immunisation"),
        "vaccines": ("vaccination", "immunization", "immunisation"),
        "tongue print": ("lingual print", "tongue pattern", "tongue biometrics", "lingual morphology"),
        "fingerprints": ("fingerprint", "dermatoglyphics"),
        "fingerprint": ("fingerprints", "dermatoglyphics"),
        "cancer": ("tumor", "neoplasm"),
        "diabetes": ("type 2 diabetes", "t2d"),
        "heart attack": ("myocardial infarction", "mi"),
        "stroke": ("cerebrovascular accident", "cva"),
        "flu": ("influenza",),
    }

    _DRIFT_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9\-]{2,}\b")
    _WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b")

    def __init__(self) -> None:
        # drift counts per claim type, token -> count
        self._drift_counts: MutableMapping[str, Dict[str, int]] = {k: {} for k in CLAIM_TYPES}
        self._lock = threading.Lock()

    # ----------------------------
    # Claim typing
    # ----------------------------
    def classify_claim(self, claim: str) -> str:
        text = (claim or "").lower()
        score = {k: 0.0 for k in CLAIM_TYPES}

        for claim_type, kws in self._CLAIM_KEYWORDS.items():
            for kw in kws:
                if kw in text:
                    score[claim_type] += 1.0

        # Strong signals
        if re.search(r"\b\d+(?:\.\d+)?\s*%?\b", text):
            score["STATISTICAL"] += 1.5
        if re.search(
            r"\b(more|less|fewer|greater|higher|lower)\b.+\bthan\b|\bvs\.?\b|\bversus\b|\bcompared to\b",
            text,
        ):
            score["NUMERIC_COMPARISON"] += 1.8
        if re.search(r"\b(population|count|number of|estimate|estimated)\b", text):
            score["NUMERIC_COMPARISON"] += 1.2
        if re.search(r"\bno two\b|\bunique\b|\bindividuality\b|\bbiometric\b", text):
            score["UNIQUENESS"] += 1.5
        if re.search(r"\bnot true\b|\bfalse\b|\bhoax\b|\bmyth\b|\bdebunk\b", text):
            score["MYTH"] += 1.5
        if re.search(r"\bbecause\b|\bvia\b|\bthrough\b|\bmechanism\b|\bpathway\b", text):
            score["MECHANISM"] += 1.0
        if re.search(r"\bsafe(?:ty)?\b|\bside effects?\b|\bharm(?:ful)?\b|\brisk\b|\binfertility\b", text):
            score["SAFETY"] += 1.2
        if re.search(r"\beffective\b|\bworks?\b|\bprevents?\b|\breduces?\b|\btreats?\b|\bcures?\b|\breverses?\b", text):
            score["EFFICACY"] += 1.2

        # If nothing matched, default to GENERAL (not MECHANISM)
        if all(v == 0.0 for v in score.values()):
            if re.search(r"\bhow many\b|\bnumber of\b|\brate of\b", text):
                return "STATISTICAL"
            return "GENERAL"

        ordered_priority = [
            "MYTH",
            "UNIQUENESS",
            "NUMERIC_COMPARISON",
            "SAFETY",
            "EFFICACY",
            "STATISTICAL",
            "MECHANISM",
            "GENERAL",
        ]
        best = max(ordered_priority, key=lambda ct: (score.get(ct, 0.0), -ordered_priority.index(ct)))
        return best if best in CLAIM_TYPES else "GENERAL"

    # ----------------------------
    # Entity extraction
    # ----------------------------
    def extract_entities(self, claim: str) -> ClaimEntities:
        text = re.sub(r"\s+", " ", claim or "").strip()
        low = text.lower()
        tokens = [t.lower() for t in self._WORD_RE.findall(text)]
        content = [t for t in tokens if t not in self._STOPWORDS and len(t) > 2]

        population = [p for p in self._POPULATION_TERMS if p in low]
        comparator = self._extract_comparator(text)
        outcome = self._extract_outcome(text)
        topic = self._extract_topic(content, text)

        anchors = self._dedupe(topic + outcome + comparator + population)
        synonyms = self._build_synonym_map(anchors)

        return ClaimEntities(
            topic=topic[:6],
            outcome=outcome[:4],
            population=population[:4],
            comparator=comparator[:3],
            anchors=anchors[:10],
            synonyms=synonyms,
        )

    # ----------------------------
    # Plan builder
    # ----------------------------
    def build_plan(self, claim: str) -> QueryPlan:
        claim_text = re.sub(r"\s+", " ", (claim or "").strip())
        claim_type = self.classify_claim(claim_text)
        entities = self.extract_entities(claim_text)
        negatives = self._collect_negatives(claim_type)
        queries = self._generate_queries(claim_text, claim_type, entities, negatives)

        notes = [
            f"classifier={claim_type}",
            f"anchors={len(entities.anchors)}",
            f"negatives={len(negatives)}",
            "adaptive_top_n=3->5->8",
            "trusted_domains_aligned_to_pse=true",
        ]
        return QueryPlan(
            claim=claim_text,
            claim_type=claim_type,
            extracted_entities=entities,
            queries=queries,
            negatives=negatives,
            strategy_notes=notes,
        )

    # ----------------------------
    # Drift learning: counted + thresholded
    # ----------------------------
    def register_drift_from_url(self, claim_type: str, url: str, title: str, snippet: str) -> None:
        claim_key = claim_type.upper().strip()
        if claim_key not in CLAIM_TYPES:
            claim_key = "GENERAL"

        raw = " ".join([(url or ""), (title or ""), (snippet or "")]).lower()
        parsed = urlparse(url or "")
        path = (parsed.path or "").lower()

        tokens: set[str] = set()
        if path.endswith(".pdf"):
            tokens.add("pdf")

        for pattern in self._LOW_SIGNAL_PATTERNS:
            if pattern in raw:
                tokens.add(pattern)

        # Whitelist which drift tokens we consider safe negatives
        for token in self._DRIFT_TOKEN_RE.findall(raw):
            if token in {"pdf", "manual", "policy", "handbook", "template", "brochure", "procurement"}:
                tokens.add(token)

        if not tokens:
            return

        with self._lock:
            bucket = self._drift_counts.setdefault(claim_key, {})
            for t in tokens:
                bucket[t] = bucket.get(t, 0) + 1
            # keep small memory
            if len(bucket) > 48:
                # prune lowest counts
                pruned = dict(sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))[:48])
                self._drift_counts[claim_key] = pruned

        logger.info(
            "[QueryDesigner] Registered drift counts for %s: %s",
            claim_key,
            {t: self._drift_counts[claim_key].get(t) for t in sorted(tokens)},
        )

    def _trusted_routes_for_claim_type(self, claim_type: str) -> List[str]:
        claim_key = (claim_type or "GENERAL").upper()
        hints = self._TRUSTED_ROUTE_HINTS_BY_TYPE.get(
            claim_key,
            self._TRUSTED_ROUTE_HINTS_BY_TYPE["GENERAL"],
        )
        routes: List[Tuple[int, str]] = []
        for domain in sorted(TRUSTED_ROOT_DOMAINS):
            try:
                idx = next(i for i, hint in enumerate(hints) if hint in domain)
            except StopIteration:
                continue
            routes.append((idx, domain))
        routes.sort(key=lambda x: (x[0], x[1]))
        ordered = [d for _, d in routes]
        return ordered or sorted(TRUSTED_ROOT_DOMAINS)

    # ----------------------------
    # Quality gate + adaptive selection
    # ----------------------------
    def quality_gate(
        self, claim_type: str, entities: ClaimEntities, url: str, title: str, snippet: str
    ) -> QualityGateResult:
        text = f"{title or ''} {snippet or ''} {url or ''}".lower()
        score = 0.0
        reasons: List[str] = []

        anchor_terms = self._expanded_anchor_terms(entities)
        if any(term in text for term in anchor_terms):
            score += 1.5
            reasons.append("anchor_match")

        type_keywords = self._CLAIM_KEYWORDS.get(claim_type, ())
        if any(k in text for k in type_keywords):
            score += 1.0
            reasons.append("claim_type_keyword")

        if is_trusted_domain(url or ""):
            score += 0.75
            reasons.append("trusted_domain")

        if any(sig in text for sig in self._LOW_SIGNAL_PATTERNS):
            score -= 1.0
            reasons.append("low_signal_penalty")

        if re.search(r"\.(?:pdf|docx?|pptx?)\b", (url or "").lower()):
            score -= 0.5
            reasons.append("file_penalty")

        passed = score >= 1.5 and ("anchor_match" in reasons or "claim_type_keyword" in reasons)
        return QualityGateResult(passed=passed, score=round(score, 3), reasons=reasons)

    def adaptive_top_n(self, claim_type: str, entities: ClaimEntities, results: Sequence[Mapping[str, str]]) -> int:
        # Decide expansion ONLY based on top-3 quality
        head = results[:3]
        passed = 0
        for item in head:
            decision = self.quality_gate(
                claim_type=claim_type,
                entities=entities,
                url=str(item.get("url", "")),
                title=str(item.get("title", "")),
                snippet=str(item.get("snippet", "")),
            )
            if decision.passed:
                passed += 1

        if passed >= 2:
            return 3
        if passed == 1:
            return 5
        return 8

    def select_results(
        self,
        claim_type: str,
        entities: ClaimEntities,
        results: Sequence[Mapping[str, str]],
    ) -> List[Dict[str, object]]:
        top_n = self.adaptive_top_n(claim_type, entities, results)

        scored: List[Tuple[float, Dict[str, object]]] = []
        for item in results[: max(top_n, 8)]:  # score only what we might take
            decision = self.quality_gate(
                claim_type=claim_type,
                entities=entities,
                url=str(item.get("url", "")),
                title=str(item.get("title", "")),
                snippet=str(item.get("snippet", "")),
            )
            enriched = dict(item)
            enriched["quality_pass"] = decision.passed
            enriched["quality_score"] = decision.score
            enriched["quality_reasons"] = decision.reasons
            scored.append((decision.score, enriched))

        scored.sort(key=lambda x: (-x[0], str(x[1].get("url", ""))))
        return [item for _, item in scored[:top_n]]

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _extract_comparator(self, text: str) -> List[str]:
        comparators: List[str] = []
        patterns = [
            r"\b(?:vs\.?|versus|compared to|instead of|in addition to)\s+([a-zA-Z0-9\-\s]{2,60})",
            r"\b([a-zA-Z0-9\-\s]{2,40})\s+\bthan\b\s+([a-zA-Z0-9\-\s]{2,40})",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                if isinstance(match, tuple):
                    parts = [self._normalize_phrase(x) for x in match]
                    comparators.extend([p for p in parts if p])
                else:
                    normalized = self._normalize_phrase(match)
                    if normalized:
                        comparators.append(normalized)
        return self._dedupe(comparators)

    def _extract_outcome(self, text: str) -> List[str]:
        outcomes: List[str] = []
        patterns = [
            (
                r"\b(?:reduce|reduces|prevent|prevents|improve|improves|increase|increases|"
                r"cause|causes|cure|cures|reverses)\s+([a-zA-Z0-9\-\s]{2,70})"
            ),
            r"\b(?:risk of|rate of|odds of|associated with)\s+([a-zA-Z0-9\-\s]{2,70})",
        ]
        for pattern in patterns:
            for raw in re.findall(pattern, text, flags=re.IGNORECASE):
                cleaned = self._normalize_phrase(re.split(r"\b(?:and|or|but|because)\b", raw, maxsplit=1)[0])
                if cleaned and len(cleaned.split()) <= 8:
                    outcomes.append(cleaned)
        return self._dedupe(outcomes)

    def _extract_topic(self, content_tokens: List[str], full_text: str) -> List[str]:
        # Exclude claim-type keywords and population terms from topic candidates
        banned = set(self._POPULATION_TERMS)
        for kws in self._CLAIM_KEYWORDS.values():
            for kw in kws:
                banned.update(self._normalize_phrase(kw).split())

        quoted = re.findall(r'"([^"]+)"', full_text)
        phrases = [self._normalize_phrase(x) for x in quoted if self._normalize_phrase(x)]
        low = full_text.lower()

        # Special n-grams for common patterns
        if "tongue print" in low:
            phrases.append("tongue print")
        if "fingerprint" in low:
            phrases.append("fingerprints")
        if "heart attack" in low:
            phrases.append("heart attack")

        freq: Dict[str, int] = {}
        for token in content_tokens:
            if token in banned or token in self._STOPWORDS:
                continue
            freq[token] = freq.get(token, 0) + 1

        ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
        tokens = [w for w, _ in ranked[:8]]

        topics = self._dedupe(phrases + tokens)
        return [t for t in topics if t and t not in self._STOPWORDS]

    def _build_synonym_map(self, anchors: Iterable[str]) -> Dict[str, List[str]]:
        synonyms: Dict[str, List[str]] = {}
        for anchor in anchors:
            key = anchor.lower().strip()
            if not key:
                continue
            values: List[str] = []
            if key in self._ENTITY_SYNONYMS:
                values.extend(self._ENTITY_SYNONYMS[key])
            if key.endswith("s") and key[:-1] in self._ENTITY_SYNONYMS:
                values.extend(self._ENTITY_SYNONYMS[key[:-1]])
            if not key.endswith("s") and f"{key}s" in self._ENTITY_SYNONYMS:
                values.extend(self._ENTITY_SYNONYMS[f"{key}s"])
            values = self._dedupe([v for v in values if v])
            if values:
                synonyms[key] = values
        return synonyms

    def _collect_negatives(self, claim_type: str) -> List[str]:
        negatives = set(self._NEGATIVE_BASE)
        negatives.update(self._NEGATIVE_BY_TYPE.get(claim_type, set()))

        # Promote drift tokens after first confirmed drift hit so the
        # next query cycle can immediately avoid the same low-signal pattern.
        with self._lock:
            bucket = self._drift_counts.get(claim_type, {})
            for token, count in bucket.items():
                if count >= 1:
                    negatives.add(token)

        # Keep small; drift will add the important stuff
        return sorted(negatives)[:16]

    def _generate_queries(
        self,
        claim: str,
        claim_type: str,
        entities: ClaimEntities,
        negatives: List[str],
    ) -> List[QuerySpec]:
        claim_compact = self._normalize_phrase(claim)
        anchors = entities.anchors or entities.topic
        if not anchors:
            anchors = [claim_compact]

        primary = self._pick_primary_anchor(anchors, claim)
        secondary = self._pick_secondary_anchor(anchors, primary)

        type_hint = self._claim_type_query_hint(claim_type)
        secondary_hint = self._claim_type_secondary_hint(claim_type)
        domain_routes = self._trusted_routes_for_claim_type(claim_type)

        candidates: List[QuerySpec] = []

        # 1) Primary evidence: quoted + unquoted (recall boost)
        candidates.append(
            QuerySpec(
                q=self._apply_negatives(f'"{primary}" {type_hint}', negatives),
                goal="primary evidence (quoted)",
                weight=1.0,
            )
        )
        candidates.append(
            QuerySpec(
                q=self._apply_negatives(f"{primary} {type_hint}", negatives),
                goal="primary evidence (unquoted)",
                weight=0.97,
            )
        )

        # 2) Dual anchor: quoted + unquoted
        candidates.append(
            QuerySpec(
                q=self._apply_negatives(f'"{primary}" "{secondary}" {type_hint}', negatives),
                goal="dual-anchor (quoted)",
                weight=0.95,
            )
        )
        candidates.append(
            QuerySpec(
                q=self._apply_negatives(f"{primary} {secondary} {type_hint}", negatives),
                goal="dual-anchor (unquoted)",
                weight=0.92,
            )
        )

        # 3) Lexical fallback
        candidates.append(
            QuerySpec(
                q=self._apply_negatives(f"{primary} {secondary} {secondary_hint}", negatives),
                goal="lexical fallback",
                weight=0.85,
            )
        )

        # 4) Trusted routing (only domains in your PSE allowlist)
        for idx, domain in enumerate(domain_routes[:4]):
            routed = self._apply_negatives(f"site:{domain} {primary} {type_hint}", negatives)
            candidates.append(
                QuerySpec(
                    q=routed,
                    goal=f"trusted routing {idx + 1}",
                    weight=max(0.70, 0.90 - (idx * 0.05)),
                )
            )

        # 5) Type-specific backups
        if claim_type == "STATISTICAL":
            candidates.append(
                QuerySpec(
                    q=self._apply_negatives(f"{primary} incidence prevalence proportion percentage", negatives),
                    goal="stats backup",
                    weight=0.82,
                )
            )
        elif claim_type == "NUMERIC_COMPARISON":
            candidates.append(
                QuerySpec(
                    q=self._apply_negatives(
                        f"{primary} {secondary} estimated count comparison world population",
                        negatives,
                    ),
                    goal="numeric comparison backup",
                    weight=0.84,
                )
            )
            candidates.append(
                QuerySpec(
                    q=self._apply_negatives(
                        f'"{primary}" "{secondary}" oral microbiome estimated oral bacteria '
                        f"human population comparison",
                        negatives,
                    ),
                    goal="numeric comparison oral microbiome backup",
                    weight=0.83,
                )
            )
        elif claim_type == "MYTH":
            candidates.append(
                QuerySpec(
                    q=self._apply_negatives(f"{primary} debunk evidence fact check", negatives),
                    goal="myth backup",
                    weight=0.82,
                )
            )
        elif claim_type == "UNIQUENESS":
            candidates.append(
                QuerySpec(
                    q=self._apply_negatives(f"{primary} biometrics individuality forensic study", negatives),
                    goal="uniqueness backup",
                    weight=0.82,
                )
            )

        # Dedup + cap to 8
        deduped: List[QuerySpec] = []
        seen: set[str] = set()
        for item in candidates:
            normalized = re.sub(r"\s+", " ", item.q.strip().lower())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(item)
            if len(deduped) >= 8:
                break

        # Ensure minimum 6 queries
        while len(deduped) < 6:
            fallback_q = self._apply_negatives(f"{primary} {secondary}", negatives)
            key = fallback_q.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(QuerySpec(q=fallback_q, goal="padding fallback", weight=0.65))
            else:
                break

        return deduped[:8]

    def _claim_type_query_hint(self, claim_type: str) -> str:
        hints = {
            "EFFICACY": "randomized trial evidence review",
            "SAFETY": "adverse effects safety evidence",
            "STATISTICAL": "statistics prevalence incidence",
            "NUMERIC_COMPARISON": "estimated count comparison population statistics",
            "MECHANISM": "mechanism pathway biological",
            "MYTH": "evidence fact check",
            "UNIQUENESS": "uniqueness individuality identification",
            "GENERAL": "evidence study",
        }
        return hints.get(claim_type, "evidence")

    def _claim_type_secondary_hint(self, claim_type: str) -> str:
        hints = {
            "EFFICACY": "systematic review meta-analysis",
            "SAFETY": "risk profile cohort",
            "STATISTICAL": "population rate registry",
            "NUMERIC_COMPARISON": "human population estimate comparative magnitude",
            "MECHANISM": "causal pathway physiology",
            "MYTH": "debunk",
            "UNIQUENESS": "forensic biometrics",
            "GENERAL": "review",
        }
        return hints.get(claim_type, "study")

    def _pick_primary_anchor(self, anchors: Sequence[str], claim: str) -> str:
        low = claim.lower()
        for anchor in anchors:
            if " " in anchor and anchor in low:
                return anchor
        return anchors[0]

    def _pick_secondary_anchor(self, anchors: Sequence[str], primary: str) -> str:
        for anchor in anchors:
            if anchor != primary:
                return anchor
        return primary

    def _apply_negatives(self, query: str, negatives: Sequence[str]) -> str:
        cleaned_query = re.sub(r"\s+", " ", query).strip()
        neg_tokens = []
        for neg in negatives[:8]:
            token = re.sub(r"[^a-z0-9\-]", "", str(neg).lower())
            if token:
                neg_tokens.append(f"-{token}")
        return f"{cleaned_query} {' '.join(neg_tokens)}".strip()

    def _expanded_anchor_terms(self, entities: ClaimEntities) -> List[str]:
        terms = list(entities.anchors)
        for key, values in entities.synonyms.items():
            terms.append(key)
            terms.extend(values)
        terms = [t.lower().strip() for t in terms if t and len(t.strip()) > 1]
        return self._dedupe(terms)

    @staticmethod
    def _normalize_phrase(raw: str) -> str:
        text = re.sub(r"\s+", " ", (raw or "").strip().lower())
        text = re.sub(r"[^a-z0-9\-\s]", "", text)
        return text.strip()

    @staticmethod
    def _dedupe(items: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in items:
            key = re.sub(r"\s+", " ", item.strip().lower())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item.strip())
        return out


_DESIGNER = CorrectiveQueryDesigner()


def build_plan(claim: str) -> QueryPlan:
    return _DESIGNER.build_plan(claim)


def register_drift_from_url(claim_type: str, url: str, title: str, snippet: str) -> None:
    _DESIGNER.register_drift_from_url(claim_type, url, title, snippet)
