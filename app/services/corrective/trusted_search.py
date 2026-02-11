import asyncio
import hashlib
import os
import re
import urllib.parse
from typing import Any, Dict, List, Tuple

import aiohttp

from app.constants.config import GOOGLE_CSE_SEARCH_URL, GOOGLE_CSE_TIMEOUT, TRUSTED_DOMAINS
from app.constants.llm_prompts import QUERY_REFORMULATION_PROMPT, REINFORCEMENT_QUERY_PROMPT
from app.core.config import settings
from app.core.logger import get_logger
from app.core.rate_limit import throttled
from app.services.common.list_ops import dedupe_list
from app.services.common.url_helpers import dedup_urls, is_accessible_url
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority
from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy

logger = get_logger(__name__)


class QuotaExceededError(Exception):
    """Raised when search API quota is exceeded."""

    pass


class TrustedSearch:
    """
    Google Custom Search (CSE) integration for trusted domain retrieval.
    Falls back to Serper.dev if Google quota is exceeded.

    Provides:
        - automated query reformulation
        - domain-filtered search results
        - async network requests
        - configurable CSE and API keys
        - Serper.dev fallback for quota issues
        - structured return format
    """

    GOOGLE_API_KEY = settings.GOOGLE_API_KEY
    GOOGLE_CSE_ID = settings.GOOGLE_CSE_ID
    SERPER_API_KEY = settings.SERPER_API_KEY

    SEARCH_URL = GOOGLE_CSE_SEARCH_URL
    SERPER_URL = "https://google.serper.dev/search"
    STRICT_TRUSTED_DOMAIN_BASE = {
        "who.int",
        "nih.gov",
        "nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "cdc.gov",
        "mayoclinic.org",
        "clevelandclinic.org",
        "health.harvard.edu",
        "medlineplus.gov",
        "fda.gov",
        "hhs.gov",
        "nhs.uk",
        "nice.org.uk",
    }
    _SERPER_SITE_BOOST_PRIORITY = [
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "nih.gov",
        "cdc.gov",
        "who.int",
        "medlineplus.gov",
        "mayoclinic.org",
        "clevelandclinic.org",
        "nhs.uk",
        "nice.org.uk",
    ]

    def __init__(
        self,
        google_api_key: str | None = None,
        google_cse_id: str | None = None,
        serper_api_key: str | None = None,
    ) -> None:
        # Resolve credentials at runtime so tests/env overrides work after import.
        self.GOOGLE_API_KEY = (
            google_api_key if google_api_key is not None else (os.getenv("GOOGLE_API_KEY") or settings.GOOGLE_API_KEY)
        )
        self.GOOGLE_CSE_ID = (
            google_cse_id if google_cse_id is not None else (os.getenv("GOOGLE_CSE_ID") or settings.GOOGLE_CSE_ID)
        )
        self.SERPER_API_KEY = (
            serper_api_key if serper_api_key is not None else (os.getenv("SERPER_API_KEY") or settings.SERPER_API_KEY)
        )

        # Allow initialization even without Google API if Serper is available
        has_google = bool(self.GOOGLE_API_KEY and self.GOOGLE_CSE_ID)
        has_serper = bool(self.SERPER_API_KEY)

        if not has_google and not has_serper:
            logger.error("No search API configured (need GOOGLE_API_KEY+CSE_ID or SERPER_API_KEY)")
            raise RuntimeError("Missing search API credentials")

        self.google_available = has_google
        self.serper_available = has_serper
        self.google_quota_exceeded = False
        self.min_allowlist_pass = max(1, int(os.getenv("TRUSTED_SEARCH_MIN_ALLOWLIST_PASS", "3")))
        extra_allowlist = {
            d.strip().lower().removeprefix("www.")
            for d in (os.getenv("TRUSTED_SEARCH_EXTRA_ALLOWLIST", "") or "").split(",")
            if d.strip()
        }
        trusted_base = {d.lower().removeprefix("www.") for d in TRUSTED_DOMAINS}
        self.strict_allowlist = set(self.STRICT_TRUSTED_DOMAIN_BASE) | trusted_base | extra_allowlist
        allowlist_fingerprint = hashlib.sha256("|".join(sorted(self.strict_allowlist)).encode("utf-8")).hexdigest()[:12]
        serper_site_env = [
            d.strip().lower().removeprefix("www.")
            for d in (os.getenv("TRUSTED_SEARCH_SERPER_SITE_DOMAINS", "") or "").split(",")
            if d.strip()
        ]
        serper_site_limit = max(1, int(os.getenv("TRUSTED_SEARCH_SERPER_SITE_MAX_DOMAINS", "4")))
        ordered_site_domains = dedupe_list(
            serper_site_env
            + self._SERPER_SITE_BOOST_PRIORITY
            + sorted(d for d in self.strict_allowlist if d not in self._SERPER_SITE_BOOST_PRIORITY)
        )
        self.serper_site_boost_domains = ordered_site_domains[:serper_site_limit]

        if has_google:
            logger.info("[TrustedSearch] Google CSE configured")
        if has_serper:
            logger.info("[TrustedSearch] Serper.dev fallback configured")
        logger.info(
            "[TrustedSearch] Trusted domain allowlist loaded: count=%d checksum=%s",
            len(self.strict_allowlist),
            allowlist_fingerprint,
        )
        logger.info(
            "[TrustedSearch] Serper site-boost domains: %s",
            self.serper_site_boost_domains,
        )

        try:
            self.llm_client = HybridLLMService()
        except Exception as e:
            logger.warning(
                "[TrustedSearch] LLM client unavailable; using deterministic/fallback query planning only: %s", e
            )
            self.llm_client = None

    # ---------------------------------------------------------------------
    # Deterministic Query Helpers
    # ---------------------------------------------------------------------
    _STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "just",
        "like",
        "every",
        "individual",
        "around",
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
        "about",
        "around",
        "average",
        "per",
        "times",
        "every",
        "each",
        "roughly",
        "approximately",
    }
    _ACTION_WORDS = {
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
        "have",
        "has",
        "had",
        "how",
        "what",
        "when",
        "where",
        "why",
        "which",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
    }
    _CONSTRAINT_KEYWORDS = [
        "statistics",
        "prevalence",
        "incidence",
        "guideline",
        "systematic review",
        "meta-analysis",
    ]
    _JUNK_QUERY_TOKENS = {
        "according",
        "significantly",
        "significant",
        "drinking",
        "improves",
        "improve",
        "every",
        "morning",
        "medical",
        "research",
        "your",
        "body",
        "produces",
        "produce",
        "produced",
        "new",
        "roughly",
        "population",
    }
    _ANCHOR_DROP_TOKENS = {
        "no",
        "not",
        "cause",
        "causes",
        "caused",
        "causing",
        "through",
        "itself",
        "scientifically",
        "supported",
        "support",
        "claim",
        "claims",
        "reported",
        "reports",
        "update",
        "updated",
        "show",
        "shows",
        "study",
        "studies",
    }
    _MERGE_PREFIXES = ("and ", "or ", "but ", "however ", "although ", "while ")

    def _expand_conjunction_fragment(self, previous_subclaim: str, fragment_subclaim: str) -> str:
        """
        Expand short conjunction fragments into standalone subclaims when possible.
        Example: "Vaccines do not cause autism" + "or the flu" ->
                 "Vaccines do not cause the flu"
        """
        prev = re.sub(r"\s+", " ", (previous_subclaim or "")).strip(" ,.").strip()
        frag = (
            re.sub(
                r"^\s*(?:and|or|but|however|although|while)\s+",
                "",
                re.sub(r"\s+", " ", (fragment_subclaim or "")).strip(),
                flags=re.IGNORECASE,
            )
            .strip(" ,.")
            .strip()
        )
        if not prev or not frag:
            return ""

        neg_causal = re.match(
            (
                r"^(?P<head>.+?\b(?:do|does|did|can|could|may|might|must|should|would|will)?\s*not\s+"
                r"(?:cause|cure|treat|prevent|trigger))\s+.+$"
            ),
            prev,
            flags=re.IGNORECASE,
        )
        if neg_causal:
            expanded = f"{neg_causal.group('head')} {frag}".strip()
            return re.sub(r"\s+", " ", expanded).strip(" ,.").strip()

        return ""

    def merge_subclaims(self, subclaims: List[str]) -> List[str]:
        """Merge conjunction fragments conservatively; preserve distinct medical subclaims."""
        merged: List[str] = []
        for raw in subclaims or []:
            s = (raw or "").strip()
            if not s:
                continue
            if not merged:
                merged.append(s)
                continue
            lower = s.lower()
            if lower.startswith(self._MERGE_PREFIXES):
                expanded = self._expand_conjunction_fragment(merged[-1], s)
                if expanded:
                    merged.append(expanded)
                    continue
                # Keep old merge behavior for list-continuation fragments.
                fragment = re.sub(r"^\s*(?:and|or|but|however|although|while)\s+", "", lower).strip()
                fragment_tokens = [t for t in self._tokenize_words(fragment) if t not in self._STOPWORDS]
                medical_tokens = {
                    "flu",
                    "influenza",
                    "autism",
                    "vaccine",
                    "vaccines",
                    "antibiotic",
                    "antibiotics",
                    "virus",
                    "viruses",
                    "sugar",
                    "hyperactivity",
                    "adhd",
                    "cholesterol",
                    "heart",
                }
                has_medical_anchor = any(t in medical_tokens for t in fragment_tokens)
                if len(fragment_tokens) <= 4:
                    if has_medical_anchor:
                        merged.append(s)
                    else:
                        merged[-1] = f"{merged[-1].rstrip(', ')} {s}".strip()
                else:
                    merged.append(s)
            else:
                merged.append(s)
        return merged

    def _extract_numbers(self, text: str) -> List[str]:
        if not text:
            return []
        nums = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)
        cleaned = []
        for n in nums:
            n = n.replace(",", "")
            if n and n not in cleaned:
                cleaned.append(n)
        return cleaned

    def _extract_numbers_raw(self, text: str) -> List[str]:
        if not text:
            return []
        nums = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)
        cleaned = []
        for n in nums:
            if n and n not in cleaned:
                cleaned.append(n)
        return cleaned

    def _tokenize_words(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", text.lower())

    def _collect_key_terms(self, texts: List[str], entities: List[str] | None = None) -> tuple[list[str], list[str]]:
        entities = entities or []
        key_terms: List[str] = []
        key_numbers: List[str] = []

        for t in texts:
            for n in self._extract_numbers(t):
                if n not in key_numbers:
                    key_numbers.append(n)
            for w in self._tokenize_words(t):
                if w in self._STOPWORDS or w in self._ACTION_WORDS or w in self._JUNK_QUERY_TOKENS:
                    continue
                if w not in key_terms:
                    key_terms.append(w)

        for e in entities:
            for w in self._tokenize_words(e):
                if w in self._STOPWORDS or w in self._ACTION_WORDS or w in self._JUNK_QUERY_TOKENS:
                    continue
                if w not in key_terms:
                    key_terms.append(w)

        return key_terms, key_numbers

    def _extract_phrase_candidates(self, text: str, entities: List[str] | None = None) -> List[str]:
        entities = entities or []
        text_l = (text or "").lower()
        tokens = [t for t in self._tokenize_words(text) if t not in self._STOPWORDS and t not in self._ACTION_WORDS]
        phrases: List[str] = []

        # Extract meaningful health patterns (better than generic bigrams).
        pattern_candidates = re.findall(
            r"\b(?:diet|intake|consumption|levels?)\s+(?:rich in|low in|high in)\s+(?:[a-zA-Z\-]+\s*){1,4}",
            text_l,
        )
        for p in pattern_candidates:
            p_tokens = self._tokenize_words(p)
            while p_tokens and p_tokens[-1] in self._ACTION_WORDS:
                p_tokens.pop()
            p = " ".join(p_tokens)
            if len(p.split()) >= 3 and p not in phrases:
                phrases.append(p)

        for p in re.findall(r"\bnoncommunicable diseases?\b", text_l):
            p = " ".join(self._tokenize_words(p))
            if p and p not in phrases:
                phrases.append(p)

        # Add non-trivial entities as exact phrases.
        for ent in entities:
            ent_clean = " ".join(self._tokenize_words(ent))
            if len(ent_clean.split()) >= 2 and ent_clean not in phrases:
                phrases.append(ent_clean)

        # Backfill with token bigrams as fallback only.
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i + 1]}"
            if bg not in phrases and len(bg) >= 8:
                phrases.append(bg)
            if len(phrases) >= 4:
                break

        return phrases[:4]

    def _sanitize_query(self, query: str) -> str:
        """Normalize query syntax and drop malformed quote-heavy fragments."""
        q = re.sub(r"\s+", " ", (query or "").strip())
        if not q:
            return ""
        # Remove unmatched quotes to avoid zero-result failures.
        if q.count('"') % 2 == 1:
            q = q.replace('"', "")
        # Keep query compact and avoid over-constraining.
        tokens = q.split()
        q = " ".join(tokens[:18]).strip()
        return q

    def _simplify_query_for_fallback(self, query: str) -> str:
        """
        Simplify operator-heavy Google query for Serper fallback.
        Serper often underperforms with strict intitle/filetype/operator chains.
        """
        q = self._sanitize_query(query)
        if not q:
            return ""
        q = re.sub(r'\bintitle:"[^"]+"\s*', "", q, flags=re.IGNORECASE)
        q = re.sub(r"\bfiletype:\w+\b", "", q, flags=re.IGNORECASE)
        q = q.replace("(", " ").replace(")", " ")
        q = re.sub(r"\bOR\b", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\s+", " ", q).strip()
        # Preserve up to two quoted anchors so fallback queries don't drift off-topic.
        # Keeping just one phrase was too lossy for multi-anchor claims.
        if q.count('"') >= 4:
            quoted = re.findall(r'"([^"]+)"', q)
            keep_parts: List[str] = []
            seen: set[str] = set()
            for phrase in quoted:
                normalized = phrase.strip()
                if not normalized:
                    continue
                dedupe_key = normalized.lower()
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                keep_parts.append(f'"{normalized}"')
                if len(keep_parts) >= 2:
                    break
            keep = " ".join(keep_parts)
            no_quotes = re.sub(r'"[^"]+"', "", q).strip()
            q = " ".join(part for part in [keep, no_quotes] if part).strip()
        return q

    def _build_phrase_variants(self, phrase: str) -> List[str]:
        """
        Build lightweight lexical variants from an input phrase without
        domain- or claim-specific hardcoded synonym dictionaries.
        """
        norm = " ".join(self._tokenize_words(phrase))
        if not norm:
            return []

        words = norm.split()
        variants = [norm]

        # Singular/plural variant on last token.
        # Keep conservative to avoid malformed forms like "diet richs".
        if words:
            last = words[-1]
            if len(words) == 1 and last.endswith("s") and len(last) > 3:
                singular = " ".join(words[:-1] + [last[:-1]])
                if singular not in variants:
                    variants.append(singular)
            elif len(words) == 1 and not last.endswith("s"):
                plural = " ".join(words[:-1] + [last + "s"])
                if plural not in variants:
                    variants.append(plural)

        # Hyphen/space normalization variant.
        if "-" in norm:
            dehyphen = norm.replace("-", " ")
            if dehyphen not in variants:
                variants.append(dehyphen)

        # Keep unique, short list for query compactness.
        out: List[str] = []
        for v in variants:
            if v not in out:
                out.append(v)
            if len(out) >= 3:
                break
        return out

    def _build_boolean_synonym_block(self, text: str, entities: List[str] | None = None) -> str:
        entities = entities or []
        phrases = self._extract_phrase_candidates(text, entities=entities)
        if not phrases:
            return ""
        best_phrase = phrases[0]
        variants = self._build_phrase_variants(best_phrase)
        if len(variants) < 2:
            return ""
        quoted = [f'"{term}"' for term in variants]
        return "(" + " OR ".join(quoted) + ")"

    def _build_negative_filters(self, text: str) -> List[str]:
        t = (text or "").lower()
        negatives: List[str] = []
        if "adult" in t or "adults" in t:
            negatives.extend(["-fetal", "-pediatric"])
        if "human" in t:
            negatives.extend(["-veterinary"])
        return negatives

    def _build_research_instruction_query(self, text: str, entities: List[str] | None = None) -> str:
        entities = entities or []
        phrases = self._extract_phrase_candidates(text, entities=entities)
        bool_block = self._build_boolean_synonym_block(text, entities=entities)
        negatives = self._build_negative_filters(text)
        numbers = self._extract_numbers_raw(text)[:2]

        parts: List[str] = []
        if bool_block:
            parts.append(bool_block)
        for p in phrases[:2]:
            parts.append(f'"{p}"')
        for n in numbers:
            parts.append(n)

        # Add one strong research/data constraint to reduce noisy results.
        text_lower = (text or "").lower()
        chosen_constraint = "statistics"
        for c in self._CONSTRAINT_KEYWORDS:
            if c in text_lower:
                chosen_constraint = c
                break
        parts.append(chosen_constraint)

        parts.extend(negatives[:2])
        return " ".join(parts).strip()

    def _build_direct_query(self, text: str, entities: List[str] | None = None) -> str:
        if not text:
            return ""
        entities = entities or []
        tokens = [
            t
            for t in self._tokenize_words(text)
            if t not in self._STOPWORDS and t not in self._ACTION_WORDS and t not in self._JUNK_QUERY_TOKENS
        ]
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        entity_tokens = set(self._tokenize_words(" ".join(entities)))

        def _score(tok: str) -> float:
            score = 0.0
            score += 2.0 * freq.get(tok, 0)
            score += min(1.0, len(tok) / 8.0)
            if tok in entity_tokens:
                score += 1.0
            return score

        ranked_terms = sorted(set(tokens), key=lambda t: (-_score(t), t))
        numbers_raw = self._extract_numbers_raw(text)
        numbers_norm = self._extract_numbers(text)
        # Avoid number-only queries; keep at most 3 numbers
        numbers_raw = numbers_raw[:3]
        numbers_norm = numbers_norm[:3]

        terms: List[str] = []
        for n in numbers_raw:
            terms.append(n)
        # include normalized number if raw had commas stripped
        for n in numbers_norm:
            if n not in terms:
                terms.append(n)
        for w in ranked_terms:
            terms.append(w)

        # Keep query concise and ensure at least 2 content words
        terms = dedupe_list(terms)
        # Ensure at least two alphabetic tokens are retained
        word_terms = [t for t in terms if t.isalpha()]
        if len(word_terms) < 2:
            for w in ranked_terms:
                if w not in terms:
                    terms.append(w)
                if len([t for t in terms if t.isalpha()]) >= 2:
                    break
        terms = terms[:8]
        return " ".join(terms).strip()

    def _extract_subclaim_anchors(self, subclaim: str, entities: List[str] | None = None) -> List[str]:
        entities = entities or []
        anchors: List[str] = []
        sub_tokens = {
            t
            for t in self._tokenize_words(subclaim)
            if t not in self._STOPWORDS
            and t not in self._ACTION_WORDS
            and t not in self._JUNK_QUERY_TOKENS
            and t not in self._ANCHOR_DROP_TOKENS
        }

        def _add(anchor: str) -> None:
            normalized = " ".join(self._tokenize_words(anchor))
            if not normalized:
                return
            norm_tokens = self._tokenize_words(normalized)
            if any(t in self._ANCHOR_DROP_TOKENS for t in norm_tokens):
                return
            if normalized not in anchors:
                anchors.append(normalized)

        # Prefer meaningful multi-word anchors first.
        for phrase in self._extract_phrase_candidates(subclaim, entities=[])[:4]:
            _add(phrase)

        for ent in entities:
            ent_tokens = [
                t
                for t in self._tokenize_words(ent)
                if t not in self._STOPWORDS and t not in self._JUNK_QUERY_TOKENS and t not in self._ANCHOR_DROP_TOKENS
            ]
            ent_n = " ".join(ent_tokens)
            if ent_n:
                # Prevent cross-subclaim entity bleed: only keep entity anchors that
                # overlap this subclaim's content.
                if not (set(ent_tokens) & sub_tokens):
                    continue
                _add(ent_n)

        freq: Dict[str, int] = {}
        for tok in self._tokenize_words(subclaim):
            if (
                tok in self._STOPWORDS
                or tok in self._ACTION_WORDS
                or tok in self._JUNK_QUERY_TOKENS
                or tok in self._ANCHOR_DROP_TOKENS
            ):
                continue
            freq[tok] = freq.get(tok, 0) + 1
        ranked_tokens = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
        for tok, _ in ranked_tokens[:8]:
            _add(tok)

        for n in self._extract_numbers_raw(subclaim):
            if n not in anchors:
                anchors.append(n)

        return anchors[:10]

    @staticmethod
    def _subclaim_intent_constraints(subclaim: str) -> List[str]:
        text = (subclaim or "").lower()

        has_quantitative = bool(
            re.search(
                r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|percent|percentage|rate|rates|statistics|prevalence|incidence)\b",
                text,
            )
        )
        has_causal = bool(
            re.search(
                (
                    r"\b(cause|causes|caused|causal|link|linked|association|associated|"
                    r"prevent|prevents|prevented|risk|increase|increases|decrease|decreases|"
                    r"reduce|reduces|cure|cures|treat|treats)\b"
                ),
                text,
            )
        )
        has_recommendation = bool(re.search(r"\b(recommend|recommended|guideline|guidelines|should|must)\b", text))
        has_negation = bool(re.search(r"\b(no|not|never|without|lack|lacks|lacking)\b", text))

        constraints: List[str] = []
        if has_quantitative:
            constraints.extend(["statistics", "prevalence", "incidence"])
        elif has_causal and has_negation:
            constraints.extend(["systematic review", "meta-analysis", "no association"])
        elif has_causal:
            constraints.extend(["systematic review", "meta-analysis", "causal association"])
        elif has_recommendation:
            constraints.extend(["guideline", "consensus statement", "clinical recommendation"])
        else:
            constraints.extend(["systematic review", "meta-analysis", "clinical study"])
        return constraints

    def _build_subclaim_anchor_queries(self, subclaim: str, entities: List[str] | None = None) -> List[str]:
        """
        Build targeted queries per subclaim with required anchors.
        This avoids token-bag drift and keeps query intent aligned.
        """
        anchors = self._extract_subclaim_anchors(subclaim, entities=entities)
        if not anchors:
            return []
        generic = {"them", "this", "that", "those", "these", "located", "studies", "study", "shows", "show"}

        def _anchor_score(a: str) -> float:
            score = 0.0
            if any(ch.isdigit() for ch in a):
                score += 4.0
            if " " in a:
                score += 1.2
            score += min(2.0, len(a) / 6.0)
            if a in generic:
                score -= 2.0
            return score

        ranked_anchors = sorted(dedupe_list(anchors), key=lambda a: (-_anchor_score(a), a))
        phrase_anchors = [a for a in ranked_anchors if " " in a and not any(ch.isdigit() for ch in a)]
        term_anchors = [a for a in ranked_anchors if " " not in a and not any(ch.isdigit() for ch in a)]
        number_anchors = [a for a in ranked_anchors if any(ch.isdigit() for ch in a)]

        core_terms: List[str] = []
        phrase_tokens: set[str] = set()
        for p in phrase_anchors[:2]:
            core_terms.append(f'"{p}"')
            phrase_tokens.update(self._tokenize_words(p))
        for t in term_anchors[:3]:
            if t in phrase_tokens:
                continue
            core_terms.append(t)
        for n in number_anchors[:2]:
            core_terms.append(n)
        if len(core_terms) < 2:
            for a in ranked_anchors:
                token = f'"{a}"' if " " in a else a
                if token not in core_terms:
                    core_terms.append(token)
                if len(core_terms) >= 2:
                    break

        constraints = self._subclaim_intent_constraints(subclaim)
        queries: List[str] = []
        for c in constraints[:3]:
            queries.append(" ".join(core_terms + [c]).strip())

        # Include one operator-style variant for high-yield biomedical sources.
        if core_terms:
            queries.append(" ".join(["site:pubmed.ncbi.nlm.nih.gov"] + core_terms + [constraints[0]]).strip())

        return dedupe_list([self._sanitize_query(x) for x in queries if x])

    def _query_quality(self, query: str, anchors: List[str]) -> Dict[str, Any]:
        q = (query or "").lower()
        q_tokens = set(self._tokenize_words(q))
        anchor_hits = 0
        for a in anchors:
            a_norm = " ".join(self._tokenize_words(a))
            if not a_norm:
                continue
            if (" " in a_norm and a_norm in q) or (a_norm in q_tokens):
                anchor_hits += 1
        junk_hits = sum(1 for t in self._tokenize_words(q) if t in self._JUNK_QUERY_TOKENS)
        return {
            "anchor_hits": anchor_hits,
            "anchor_total": len(anchors),
            "junk_hits": junk_hits,
            "token_count": len(self._tokenize_words(q)),
        }

    def _passes_query_quality(self, query: str, anchors: List[str]) -> bool:
        quality = self._query_quality(query, anchors)
        if quality["junk_hits"] > 1:
            return False
        if quality["token_count"] < 3:
            return False
        if quality["anchor_total"] > 0 and quality["anchor_hits"] == 0:
            return False
        return True

    def _build_direct_queries(self, subclaims: List[str], entities: List[str] | None = None) -> List[str]:
        entities = entities or []
        queries: List[str] = []
        for sub in subclaims:
            q = self._build_direct_query(sub, entities=entities)
            if q:
                queries.append(q)
        return dedupe_list(queries)

    def _query_has_key_terms(self, query: str, key_terms: List[str], key_numbers: List[str]) -> bool:
        if not query:
            return False
        q = query.lower()
        q_num = q.replace(",", "")
        for n in key_numbers:
            if n and n in q_num:
                return True
        for term in key_terms:
            if term and re.search(rf"\b{re.escape(term)}\b", q):
                return True
        return False

    def _filter_queries(self, queries: List[str], key_terms: List[str], key_numbers: List[str]) -> List[str]:
        filtered: List[str] = []
        generic_suffixes = {
            "medical evidence",
            "scientific research",
            "clinical verification",
            "factual analysis",
        }
        for q in queries:
            if not isinstance(q, str):
                continue
            q = q.strip().lower()
            if not q:
                continue
            if any(q.endswith(sfx) for sfx in generic_suffixes):
                continue
            if not self._query_has_key_terms(q, key_terms, key_numbers):
                continue
            filtered.append(q)
        return dedupe_list(filtered)

    # ---------------------------------------------------------------------
    # Query Reformulation
    # ---------------------------------------------------------------------
    def _build_question_queries(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = set(self._tokenize_words(text))
        queries: List[str] = []

        if "bones" in tokens or "bone" in tokens:
            if "hands" in tokens or "hand" in tokens or "feet" in tokens or "foot" in tokens:
                queries.append("how many bones are in adult human hands and feet")
                queries.append("how many bones are in the hands and feet")
            if "adult" in tokens or "adults" in tokens:
                queries.append("how many bones does an adult human have")
            else:
                queries.append("how many bones does a human have")

        if "heart" in tokens and ("beats" in tokens or "beat" in tokens):
            if "day" in tokens or "daily" in tokens:
                queries.append("how many times does the human heart beat per day")

        if "cells" in tokens or "cell" in tokens:
            if "second" in tokens or "seconds" in tokens:
                queries.append("how many new cells does the human body produce per second")

        if "tongue" in tokens and ("print" in tokens or "prints" in tokens):
            queries.append("are tongue prints unique")
            queries.append("unique tongue prints individuals")

        return dedupe_list(queries)

    def _build_advanced_operator_queries(self, subclaim: str, entities: List[str] | None = None) -> List[str]:
        entities = entities or []
        base = self._build_research_instruction_query(subclaim, entities=entities)
        if not base:
            return []

        adv = [base]
        # Research-paper style query.
        adv.append(f"{base} filetype:pdf")
        # Strong title constraint variant.
        adv.append(f'intitle:"meta-analysis" {base}')
        return dedupe_list(adv)

    def _build_domain_specific_queries(self, text: str) -> List[str]:
        """
        Build generic claim-level boosters from extracted phrases/anchors.
        No claim-specific hardcoded templates.
        """
        if not text:
            return []

        anchors = self._extract_subclaim_anchors(text, entities=[])
        if not anchors:
            return []

        phrase_anchors = [a for a in anchors if " " in a and not any(ch.isdigit() for ch in a)]
        term_anchors = [a for a in anchors if " " not in a and not any(ch.isdigit() for ch in a)]
        number_anchors = [a for a in anchors if any(ch.isdigit() for ch in a)]

        core_terms: List[str] = []
        for p in phrase_anchors[:2]:
            core_terms.append(f'"{p}"')
        core_terms.extend(term_anchors[:3])
        core_terms.extend(number_anchors[:1])
        core_terms = core_terms[:6]
        if not core_terms:
            return []

        constraints = self._subclaim_intent_constraints(text)
        queries = [
            " ".join(core_terms + [constraints[0]]).strip(),
            " ".join(core_terms + [constraints[1]]).strip() if len(constraints) > 1 else "",
        ]
        return dedupe_list([self._sanitize_query(q) for q in queries if q])

    async def reformulate_queries(
        self,
        text: str,
        failed_entities: List[str],
        entities: List[str] | None = None,
        subclaims: List[str] | None = None,
    ) -> List[str]:
        """
        LLM-powered search query reformulation.
        Produces highly optimized evidence retrieval phrases.
        """
        entities = entities or []
        subclaims = subclaims or []

        direct_query = self._build_direct_query(text, entities=entities)
        key_terms, key_numbers = self._collect_key_terms(texts=[text] + subclaims, entities=entities)

        if self.llm_client is None:
            logger.info("[TrustedSearch] LLM unavailable; using fallback reformulation only")
            fallback = self._fallback_queries(text, failed_entities)
            fallback = [self._sanitize_query(q) for q in fallback if q]
            merged = [direct_query] + fallback if direct_query else fallback
            filtered = self._filter_queries(merged, key_terms, key_numbers)
            return dedupe_list(filtered)

        prompt = f"""
{QUERY_REFORMULATION_PROMPT}

POST TEXT:
{text}

SUBCLAIMS:
{subclaims}

ENTITIES:
{entities}

FAILED ENTITIES:
{failed_entities}
"""

        try:
            # HIGH priority: Query reformulation is crucial for good search results
            result = await self.llm_client.ainvoke(
                prompt,
                response_format="json",
                priority=LLMPriority.HIGH,
                call_tag="query_reformulation",
            )
            queries = result.get("queries", [])
            cleaned = [self._sanitize_query(q.strip().lower()) for q in queries if isinstance(q, str)]
            cleaned = [q for q in cleaned if q]
            merged = [direct_query] + cleaned if direct_query else cleaned
            filtered = self._filter_queries(merged, key_terms, key_numbers)
            return dedupe_list(filtered)  # dedupe but preserve order

        except Exception as e:
            logger.error(f"[TrustedSearch] LLM query reformulation failed: {e}")
            # fallback -> old heuristic (plus direct query)
            fallback = self._fallback_queries(text, failed_entities)
            fallback = [self._sanitize_query(q) for q in fallback if q]
            merged = [direct_query] + fallback if direct_query else fallback
            filtered = self._filter_queries(merged, key_terms, key_numbers)
            return dedupe_list(filtered)

    # ---------------------------------------------------------------------
    # Fallback Query Generation
    # ---------------------------------------------------------------------
    def _fallback_queries(self, text: str, failed_entities: List[str]) -> List[str]:
        base = text.lower()
        queries = [
            f"{base} medical evidence",
            f"{base} scientific research",
            f"{base} clinical verification",
            f"{base} factual analysis",
        ]

        for ent in failed_entities or []:
            queries.append(f"{ent} medical evidence")
            queries.append(f"{ent} health research")
            queries.append(f"{ent} scientific facts")

        return dedupe_list(queries)

    # ---------------------------------------------------------------------
    # Single Query Search (Google CSE)
    # ---------------------------------------------------------------------
    @throttled(limit=100, period=60.0, name="google_cse")
    async def search_query_google(self, session: aiohttp.ClientSession, query: str) -> Tuple[List[str], bool]:
        """
        Runs a single Google CSE request and returns trusted URLs.
        Returns (urls, quota_exceeded) tuple.
        """
        import urllib.parse

        encoded_query = urllib.parse.quote_plus(query)

        url = self.SEARCH_URL.format(
            key=self.GOOGLE_API_KEY,
            cse=self.GOOGLE_CSE_ID,
            query=encoded_query,
        )

        try:
            async with session.get(url, timeout=GOOGLE_CSE_TIMEOUT) as resp:
                data = await resp.json()

                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    logger.error(f"[TrustedSearch:Google] API error: {error_msg}")

                    if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                        return [], True

                    return [], False

                if "items" not in data:
                    search_info = data.get("searchInformation", {})
                    total = search_info.get("totalResults", "0")
                    logger.warning(f"[TrustedSearch:Google] No items for '{query}' (total={total})")
                    return [], False

                all_urls = [item.get("link", "N/A") for item in data["items"]]
                logger.info(f"[TrustedSearch:Google] '{query}' raw: {all_urls[:3]}...")

                items = [
                    {
                        "link": item.get("link"),
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                    }
                    for item in data["items"]
                ]
                urls, _ = self._filter_trusted_urls(items, provider="Google", query=query)
                logger.info(f"[TrustedSearch:Google] '{query}' -> {len(urls)}/{len(data['items'])} trusted")
                return urls, False

        except asyncio.TimeoutError:
            logger.error(f"[TrustedSearch:Google] Timeout for '{query}'")
            return [], False
        except Exception as e:
            logger.error(f"[TrustedSearch:Google] Failed for '{query}': {e}")
            return [], False

    # ---------------------------------------------------------------------
    # Single Query Search (Serper.dev fallback)
    # ---------------------------------------------------------------------
    async def search_query_serper(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """
        Runs a single Serper.dev search request and returns trusted URLs.
        Serper provides 2,500 free searches/month.
        """
        if not self.SERPER_API_KEY:
            return []

        headers = {
            "X-API-KEY": self.SERPER_API_KEY,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "num": 10,  # Get 10 results
        }

        try:
            async with session.post(
                self.SERPER_URL,
                headers=headers,
                json=payload,
                timeout=GOOGLE_CSE_TIMEOUT,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[TrustedSearch:Serper] HTTP {resp.status}: {error_text[:200]}")
                    return []

                data = await resp.json()

                organic = data.get("organic", [])
                if not organic:
                    logger.warning(f"[TrustedSearch:Serper] No results for '{query}'")
                    return []

                all_urls = [r.get("link", "N/A") for r in organic]
                logger.info(f"[TrustedSearch:Serper] '{query}' raw: {all_urls[:3]}...")

                items = [
                    {
                        "link": result.get("link"),
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                    }
                    for result in organic
                ]
                urls, _ = self._filter_trusted_urls(items, provider="Serper", query=query)
                logger.info(f"[TrustedSearch:Serper] '{query}' -> {len(urls)}/{len(organic)} trusted")
                return urls

        except asyncio.TimeoutError:
            logger.error(f"[TrustedSearch:Serper] Timeout for '{query}'")
            return []
        except Exception as e:
            logger.error(f"[TrustedSearch:Serper] Failed for '{query}': {e}")
            return []

    async def search_query_pubmed(self, session: aiohttp.ClientSession, query: str, max_results: int = 8) -> List[str]:
        """
        PubMed fallback via NCBI E-utilities when Google/Serper produce weak results.
        Returns PubMed article URLs from PMID list.
        """
        try:
            term = urllib.parse.quote_plus(query)
            url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&retmode=json&retmax={max_results}&sort=relevance&term={term}"
            )
            async with session.get(url, timeout=GOOGLE_CSE_TIMEOUT) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
            id_list = ((data.get("esearchresult") or {}).get("idlist") or [])[:max_results]
            urls = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in id_list if pmid]
            logger.info(f"[TrustedSearch:PubMed] '{query}' -> {len(urls)} pmids")
            return urls
        except Exception as e:
            logger.warning(f"[TrustedSearch:PubMed] Failed for '{query}': {e}")
            return []

    # ---------------------------------------------------------------------
    # Unified Search (with fallback)
    # ---------------------------------------------------------------------
    async def search_query(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """
        Search with automatic fallback: Google CSE -> Serper.dev
        """
        query = self._sanitize_query(query)
        if not query:
            return []

        # If Google quota already exceeded, go straight to Serper
        if self.google_quota_exceeded:
            if self.serper_available:
                serper = await self.search_query_serper_with_site_boost(session, query)
                if serper:
                    return serper
                return await self.search_query_pubmed(session, query)
            return []

        # Try Google first
        if self.google_available:
            urls, quota_exceeded = await self.search_query_google(session, query)

            if quota_exceeded:
                self.google_quota_exceeded = True
                logger.warning("[TrustedSearch] Google CSE quota exceeded! " "Switching to Serper.dev fallback...")

                if self.serper_available:
                    serper = await self.search_query_serper_with_site_boost(session, query)
                    if serper:
                        return serper
                    return await self.search_query_pubmed(session, query)
                else:
                    logger.error(
                        "[TrustedSearch] No SERPER_API_KEY configured. "
                        "Set SERPER_API_KEY env var for fallback search."
                    )
                    return []

            if len(urls) >= self.min_allowlist_pass:
                return urls
            logger.info(
                "[TrustedSearch] Google trusted results below threshold (%d < %d), merging Serper fallback",
                len(urls),
                self.min_allowlist_pass,
            )
            if self.serper_available:
                serper_urls = await self.search_query_serper_with_site_boost(session, query)
                if serper_urls:
                    merged = dedupe_list(urls + serper_urls)
                    return merged
            if urls:
                return urls
            return await self.search_query_pubmed(session, query)

        # No Google, try Serper directly
        if self.serper_available:
            serper = await self.search_query_serper_with_site_boost(session, query)
            if serper:
                return serper
            return await self.search_query_pubmed(session, query)

        return []

    # ---------------------------------------------------------------------
    # Domain Whitelisting
    # ---------------------------------------------------------------------
    def is_trusted(self, url: str) -> bool:
        """
        Returns True if domain is in strict trusted domain allowlist.
        """
        if not is_accessible_url(url):
            return False
        try:
            parsed = urllib.parse.urlparse(url)
            domain = (parsed.netloc or "").lower()
            if "@" in domain:
                domain = domain.split("@")[-1]
            if ":" in domain:
                domain = domain.split(":")[0]
            domain = domain.removeprefix("www.")
            if not domain:
                return False
            if domain in self.strict_allowlist:
                return True

            for trusted in self.strict_allowlist:
                if domain == trusted or domain.endswith("." + trusted):
                    return True

            return False
        except Exception:
            return False

    @staticmethod
    def _is_low_signal_url(url: str) -> bool:
        u = (url or "").lower()
        low_signal_patterns = [
            "/references",
            "human-verification",
            "javascript-and-cookies",
            "/study/nct",  # trial registry pages often contain template metadata only
        ]
        return any(p in u for p in low_signal_patterns)

    def _url_quality_reject_reason(self, url: str, title: str = "", snippet: str = "") -> str | None:
        u = (url or "").strip()
        low = u.lower()
        if not u:
            return "empty_url"
        if "data:text" in low:
            return "data_text_scheme"
        if "pano=" in low:
            return "pano_param"
        if re.search(r"[A-Za-z0-9+/]{32,}={0,2}", u):
            return "base64_blob"
        if self._is_low_signal_url(u):
            return "low_signal_pattern"
        parsed = urllib.parse.urlparse(u)
        query = parsed.query or ""
        params = urllib.parse.parse_qsl(query, keep_blank_values=True)
        if len(query) > 450 or len(params) > 12:
            return "querystring_too_long"
        if params and all(k.lower().startswith(("utm_", "gclid", "fbclid", "mc_")) for k, _ in params):
            return "tracking_only"
        path = (parsed.path or "").strip("/")
        shell_paths = {"", "index", "home", "search", "category", "tags", "about", "contact", "sitemap"}
        if path.lower() in shell_paths and not (title or snippet):
            return "site_shell_page"
        return None

    def _filter_trusted_urls(
        self, items: List[Dict[str, Any]], provider: str, query: str
    ) -> Tuple[List[str], Dict[str, int]]:
        urls: List[str] = []
        rejected_allowlist = 0
        rejected_quality = 0
        for item in items:
            link = item.get("link")
            if not link:
                continue
            if not self.is_trusted(link):
                rejected_allowlist += 1
                continue
            reason = self._url_quality_reject_reason(link, item.get("title", ""), item.get("snippet", ""))
            if reason is not None:
                rejected_quality += 1
                continue
            urls.append(link)

        logger.info(
            "[TrustedSearch:%s] query='%s' count_rejected_by_allowlist=%d count_rejected_by_quality=%d "
            "final_trusted_count=%d",
            provider,
            query,
            rejected_allowlist,
            rejected_quality,
            len(urls),
        )
        return urls, {
            "count_rejected_by_allowlist": rejected_allowlist,
            "count_rejected_by_quality": rejected_quality,
            "final_trusted_count": len(urls),
        }

    # ---------------------------------------------------------------------
    # Site-Specific Query Generation
    # ---------------------------------------------------------------------
    def _generate_site_queries(self, base_query: str, max_sites: int = 4) -> List[str]:
        """
        Generate queries with site: operator for TOP trusted domains only.
        Kept small for speed and quota efficiency.
        """
        # Prioritize high-yield medical sources (capped to 3-4 sites)
        priority_sites = [
            "medlineplus.gov",
            "cdc.gov",
            "nih.gov",
            "who.int",
            "pubmed.ncbi.nlm.nih.gov",
        ][:max_sites]

        site_queries = []
        for site in priority_sites:
            if base_query:
                site_queries.append(f"site:{site} {base_query}")

        return site_queries

    def _build_serper_site_boost_queries(self, query: str) -> List[str]:
        """
        Build site-scoped Serper queries anchored to trusted domains.
        """
        q = self._sanitize_query(query)
        if not q:
            return []
        if "site:" in q.lower():
            return [q]
        domains = getattr(self, "serper_site_boost_domains", []) or self._SERPER_SITE_BOOST_PRIORITY[:4]
        return [f"site:{domain} {q}" for domain in domains if domain]

    async def search_query_serper_with_site_boost(self, session: aiohttp.ClientSession, query: str) -> List[str]:
        """
        Serper fallback with trusted-domain site: boosts when generic results are weak.
        """
        simplified = self._simplify_query_for_fallback(query)
        if not simplified:
            return []

        base_urls = await self.search_query_serper(session, simplified)
        if len(base_urls) >= self.min_allowlist_pass or "site:" in simplified.lower():
            return base_urls

        boosted_urls: List[str] = []
        for site_query in self._build_serper_site_boost_queries(simplified):
            site_urls = await self.search_query_serper(session, site_query)
            if site_urls:
                boosted_urls = dedupe_list(boosted_urls + site_urls)
            merged_count = len(dedupe_list(base_urls + boosted_urls))
            if merged_count >= self.min_allowlist_pass:
                break

        merged = dedupe_list(base_urls + boosted_urls)
        logger.info(
            "[TrustedSearch] Serper site-boost merged results: base=%d boosted=%d merged=%d",
            len(base_urls),
            len(boosted_urls),
            len(merged),
        )
        return merged

    # ---------------------------------------------------------------------
    # Generate All Queries (without executing search)
    # ---------------------------------------------------------------------
    async def generate_search_queries(
        self,
        post_text: str,
        failed_entities: List[str],
        max_queries: int = 6,
        subclaims: List[str] | None = None,
        entities: List[str] | None = None,
    ) -> List[str]:
        """
        Generate all search queries upfront without executing any search API calls.
        This allows the pipeline to control query execution one-by-one for quota optimization.

        Returns:
            List of search queries (direct subclaim + site-specific + LLM-reformulated)
        """
        entities = entities or []

        # 1) Determine subclaims and merge fragments for better query quality
        if subclaims is None:
            subclaims = AdaptiveTrustPolicy().decompose_claim(post_text)
        merged_subclaims = self.merge_subclaims(subclaims or [])

        key_terms, key_numbers = self._collect_key_terms(texts=[post_text] + merged_subclaims, entities=entities)

        # 2) Subclaim-first deterministic generation with anchor constraints.
        direct_queries: List[str] = []
        advanced_queries: List[str] = []
        question_queries: List[str] = []
        anchor_queries: List[str] = []
        required_per_subclaim: List[str] = []

        for idx, sub in enumerate(merged_subclaims):
            anchors = self._extract_subclaim_anchors(sub, entities=entities)
            required_query_for_subclaim = ""
            sub_anchor_queries = self._build_subclaim_anchor_queries(sub, entities=entities)
            for aq in sub_anchor_queries:
                if self._passes_query_quality(aq, anchors):
                    if not required_query_for_subclaim:
                        required_query_for_subclaim = aq
                    anchor_queries.append(aq)
                    qmeta = self._query_quality(aq, anchors)
                    logger.info(
                        "[TrustedSearch][QueryQuality] subclaim=%d query='%s' anchors=%d/%d junk=%d tokens=%d",
                        idx + 1,
                        aq,
                        qmeta["anchor_hits"],
                        qmeta["anchor_total"],
                        qmeta["junk_hits"],
                        qmeta["token_count"],
                    )
            q = self._build_direct_query(sub, entities=entities)
            if q and self._passes_query_quality(q, anchors):
                if not required_query_for_subclaim:
                    required_query_for_subclaim = q
                direct_queries.append(q)
            for aq in self._build_advanced_operator_queries(sub, entities=entities):
                if self._passes_query_quality(aq, anchors):
                    advanced_queries.append(aq)
            q_questions = self._build_question_queries(sub)
            if q_questions:
                candidate = q_questions[0]
                if self._passes_query_quality(candidate, anchors):
                    question_queries.append(candidate)
                    if not required_query_for_subclaim:
                        required_query_for_subclaim = candidate
            if required_query_for_subclaim:
                required_per_subclaim.append(self._sanitize_query(required_query_for_subclaim))

        direct_queries = self._filter_queries(direct_queries, key_terms, key_numbers)
        advanced_queries = self._filter_queries(advanced_queries, key_terms, key_numbers)
        question_queries = self._filter_queries(question_queries, key_terms, key_numbers)
        anchor_queries = self._filter_queries(anchor_queries, key_terms, key_numbers)
        domain_queries = self._filter_queries(self._build_domain_specific_queries(post_text), key_terms, key_numbers)

        # 3) Add LLM reformulated queries early, but only if they pass strict filters.
        llm_queries: List[str] = []
        deterministic_pool = dedupe_list(anchor_queries + direct_queries + advanced_queries + domain_queries)
        if len(deterministic_pool) < max_queries:
            llm_queries = await self.reformulate_queries(
                post_text,
                failed_entities,
                entities=entities,
                subclaims=merged_subclaims,
            )
            llm_queries = [self._sanitize_query(q) for q in llm_queries if q]
            llm_queries = self._filter_queries(llm_queries, key_terms, key_numbers)
        else:
            logger.info(
                "[TrustedSearch] Skipping LLM reformulation (deterministic queries already sufficient: %d)",
                len(deterministic_pool),
            )

        # 4) Site-specific variants for top direct/LLM/operator queries
        site_queries: List[str] = []
        site_bases = []
        site_bases.extend(anchor_queries[:1])
        site_bases.extend(direct_queries[:1])
        site_bases.extend(llm_queries[:1])
        site_bases.extend(advanced_queries[:1])
        for q in dedupe_list(site_bases):
            site_queries.extend(self._generate_site_queries(q))

        all_queries = dedupe_list(
            anchor_queries
            + direct_queries
            + advanced_queries
            + domain_queries
            + llm_queries
            + question_queries
            + site_queries
        )

        # Final validation (strictly direct to claim terms) and budget cap
        all_queries = [self._sanitize_query(q) for q in all_queries if q]
        all_queries = self._filter_queries(all_queries, key_terms, key_numbers)
        all_queries = dedupe_list(all_queries)
        required_filtered: List[str] = []
        for rq in required_per_subclaim:
            if not rq:
                continue
            rq_clean = self._sanitize_query(rq)
            if not rq_clean:
                continue
            if not self._query_has_key_terms(rq_clean, key_terms, key_numbers):
                continue
            if rq_clean not in required_filtered:
                required_filtered.append(rq_clean)
        # Ensure query planner retains at least one query per detected subclaim when possible.
        all_queries = required_filtered + [q for q in all_queries if q not in required_filtered]
        all_queries = all_queries[:max_queries]
        for i, q in enumerate(all_queries):
            qmeta = self._query_quality(q, key_terms)
            logger.info(
                "[TrustedSearch][QueryQuality][Final] idx=%d query='%s' anchors=%d/%d junk=%d tokens=%d",
                i + 1,
                q,
                qmeta["anchor_hits"],
                qmeta["anchor_total"],
                qmeta["junk_hits"],
                qmeta["token_count"],
            )
        logger.info(
            "[TrustedSearch][QueryQuality][Summary] subclaims=%d anchor=%d direct=%d advanced=%d "
            "domain=%d llm=%d final=%d",
            len(merged_subclaims),
            len(anchor_queries),
            len(direct_queries),
            len(advanced_queries),
            len(domain_queries),
            len(llm_queries),
            len(all_queries),
        )
        logger.info(f"[TrustedSearch] Generated {len(all_queries)} queries for quota-optimized search")

        return all_queries

    # ---------------------------------------------------------------------
    # Execute Single Query (quota-optimized)
    # ---------------------------------------------------------------------
    async def execute_single_query(self, query: str) -> List[str]:
        """
        Execute a single search query and return trusted URLs.
        This is the quota-optimized method - call only when needed.

        Returns:
            List of trusted URLs from this single query
        """
        logger.info(f"[TrustedSearch] Executing single query: '{query}'")

        async with aiohttp.ClientSession() as session:
            try:
                urls = await self.search_query(session, query)
                logger.info(f"[TrustedSearch] Single query returned {len(urls)} trusted URLs")
                return urls
            except Exception as e:
                logger.error(f"[TrustedSearch] Single query failed: {e}")
                return []

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Execute a search query and return results formatted as dicts with 'url' key.
        Used by VerdictGenerator for web evidence fetching.

        Returns:
            List of dicts: [{"url": "https://example.com"}, ...]
        """
        urls = await self.execute_single_query(query)
        # Limit results and format as expected by caller
        return [{"url": url} for url in urls[:max_results]]

    # ---------------------------------------------------------------------
    # Main Search Handler (legacy - runs all queries)
    # ---------------------------------------------------------------------
    async def run(
        self,
        post_text: str,
        failed_entities: List[str],
        min_urls: int = 3,
        max_queries: int = 15,
    ) -> List[str]:
        """
        Executes reformulation → sequential Google CSE queries → URL dedupe.
        Stops early once min_urls threshold is reached.
        """
        # 1) Generate reformulated queries
        queries = await self.reformulate_queries(post_text, failed_entities)
        logger.info(f"[TrustedSearch] Reformulated queries: {queries}")

        # 2) Also generate site-specific queries for better targeting
        if queries:
            # Use first query as base for site-specific searches
            site_queries = self._generate_site_queries(queries[0])
            # Interleave: original, site-specific, original, site-specific...
            all_queries = []
            for i, q in enumerate(queries):
                all_queries.append(q)
                if i < len(site_queries):
                    all_queries.append(site_queries[i])
            # Add remaining site queries
            all_queries.extend(site_queries[len(queries) :])
        else:
            all_queries = queries

        # Limit total queries
        all_queries = all_queries[:max_queries]
        logger.info(f"[TrustedSearch] Total queries to try: {len(all_queries)}")

        # 3) Execute queries sequentially until we have enough URLs
        collected_urls: set[str] = set()

        async with aiohttp.ClientSession() as session:
            for i, query in enumerate(all_queries):
                logger.info(f"[TrustedSearch] Executing query {i + 1}/{len(all_queries)}: '{query}'")

                try:
                    urls = await self.search_query(session, query)
                    collected_urls.update(urls)

                    logger.info(f"[TrustedSearch] Progress: {len(collected_urls)} URLs " f"(need {min_urls})")

                    # Early exit if we have enough
                    if len(collected_urls) >= min_urls:
                        logger.info(
                            f"[TrustedSearch] Threshold reached ({len(collected_urls)} >= "
                            f"{min_urls}), stopping search"
                        )
                        break

                except Exception as e:
                    logger.warning(f"[TrustedSearch] Query '{query}' failed: {e}")
                    continue

        # 4) Dedupe and return
        urls = dedup_urls(list(collected_urls))
        logger.info(f"[TrustedSearch] Final result: {len(urls)} trusted URLs")
        return list(urls)

    async def google_search(self, post_text: str, failed_entities: List[str] | None = None) -> List[str]:
        """
        Alias for run() - performs Google CSE search with query reformulation.
        """
        if failed_entities is None:
            failed_entities = []
        return await self.run(post_text, failed_entities)

    # ---------------------------------------------------------------------
    # LLM-powered Reinforced Query Generation
    # ---------------------------------------------------------------------
    async def llm_reformulate_for_reinforcement(
        self,
        low_conf_items: List[Dict[str, Any]],
        failed_entities: List[str],
    ) -> List[str]:
        """
        Uses LLM (Groq) to generate highly optimized reinforcement search queries.
        Much better than heuristic string concatenation.
        """

        base_statements = [item.get("statement", "") for item in low_conf_items]
        base_entities = failed_entities or []

        prompt = REINFORCEMENT_QUERY_PROMPT.format(statements=base_statements, entities=base_entities)

        try:
            # HIGH priority: Reinforcement query generation is crucial for finding evidence
            result = await self.llm_client.ainvoke(
                prompt,
                response_format="json",
                priority=LLMPriority.HIGH,
                call_tag="query_reformulation",
            )
            queries = result.get("queries", [])

            # safety: ensure list[str]
            return [q for q in queries if isinstance(q, str) and q.strip()]

        except Exception as e:
            logger.warning(f"[TrustedSearch] LLM reinforcement query generation failed: {e}")

            # fallback to simple heuristics
            fallback = []
            for stmt in base_statements:
                fallback.append(f"{stmt} peer reviewed research")
                fallback.append(f"{stmt} scientific study NIH CDC")

            for ent in base_entities:
                fallback.append(f"{ent} medical research")
                fallback.append(f"{ent} clinical facts verified")

            return dedupe_list(fallback)

    # ---------------------------------------------------------------------
    # Reinforced Search for Low Confidence Cases
    # ---------------------------------------------------------------------
    async def reinforce_search(
        self,
        low_conf_items: List[Dict[str, Any]],
        failed_entities: List[str],
        max_queries: int = 8,
    ) -> List[str]:
        """
        LLM-powered reinforcement search using Kimi/Groq.
        Calls Google CSE on the generated queries.
        """

        # 1) Generate smarter queries
        queries = await self.llm_reformulate_for_reinforcement(low_conf_items, failed_entities)
        queries = queries[:max_queries]  # cap

        logger.info(f"[TrustedSearch] Reinforcement queries: {queries}")

        # 2) Perform Google CSE search
        all_urls = []
        async with aiohttp.ClientSession() as session:
            for q in queries:
                try:
                    urls = await self.search_query(session, q)
                    all_urls.extend(urls)
                except Exception as e:
                    logger.warning(f"[TrustedSearch] Reinforcement search failed for '{q}': {e}")

        # 3) Deduplicate
        return dedup_urls(all_urls)
