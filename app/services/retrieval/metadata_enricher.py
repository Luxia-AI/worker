import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from app.core.logger import get_logger
from app.services.common.url_helpers import extract_domain
from app.services.llms.hybrid_service import HybridLLMService, LLMPriority

logger = get_logger(__name__)


TOPIC_TAXONOMY = [
    "anatomy",
    "physiology",
    "pathology",
    "pharmacology",
    "epidemiology",
    "genetics",
    "public_health",
    "nutrition",
    "other",
]

_TOPIC_KEYWORDS: Dict[str, set[str]] = {
    "anatomy": {
        "bone",
        "bones",
        "skeleton",
        "muscle",
        "tongue",
        "heart",
        "hand",
        "hands",
        "foot",
        "feet",
        "organ",
        "organs",
        "anatomy",
        "structure",
    },
    "physiology": {
        "beats",
        "beat",
        "rate",
        "blood",
        "cells",
        "cell",
        "breathing",
        "respiration",
        "metabolism",
        "function",
    },
    "pathology": {
        "disease",
        "disorder",
        "syndrome",
        "cancer",
        "infection",
        "tumor",
        "symptom",
    },
    "pharmacology": {"drug", "medication", "dose", "dosing", "therapy", "treatment"},
    "epidemiology": {"incidence", "prevalence", "risk", "cohort", "case", "cases", "rate"},
    "genetics": {"gene", "genes", "mutation", "genetic", "chromosome", "allele"},
    "public_health": {"guideline", "vaccination", "population", "policy", "public health", "cdc", "who"},
    "nutrition": {"diet", "calorie", "vitamin", "nutrient", "nutrition", "food"},
}

_DOMAIN_TOPIC_HINTS: Dict[str, List[str]] = {
    "medlineplus.gov": ["anatomy", "physiology"],
    "nlm.nih.gov": ["anatomy", "physiology"],
    "pubmed.ncbi.nlm.nih.gov": ["pathology", "pharmacology", "epidemiology"],
    "cdc.gov": ["public_health", "epidemiology"],
    "who.int": ["public_health", "epidemiology"],
    "nih.gov": ["pathology", "genetics"],
}


class TopicClassifier:
    def __init__(self) -> None:
        self.llm = HybridLLMService()
        self._cache: Dict[str, Tuple[List[str], float]] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", text.lower())

    def _rule_classify(self, statement: str, entities: List[str], domain: str | None) -> Tuple[List[str], float]:
        tokens = set(self._tokenize(statement))
        for ent in entities or []:
            tokens.update(self._tokenize(ent))

        scores: Dict[str, int] = {t: 0 for t in TOPIC_TAXONOMY if t != "other"}

        if domain:
            for hint_domain, topics in _DOMAIN_TOPIC_HINTS.items():
                if domain.endswith(hint_domain):
                    for t in topics:
                        scores[t] += 2

        for topic, keywords in _TOPIC_KEYWORDS.items():
            overlap = len(tokens & keywords)
            scores[topic] += overlap

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if not ranked or ranked[0][1] == 0:
            return [], 0.0

        best_topic, best_score = ranked[0]
        confidence = min(1.0, best_score / 5.0)
        return [best_topic], confidence

    async def classify(self, statement: str, entities: List[str], source_url: str | None) -> Tuple[List[str], float]:
        cache_key = f"{statement}|{','.join(entities)}|{source_url}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        domain = extract_domain(source_url or "") or ""
        topics, confidence = self._rule_classify(statement, entities, domain)
        if topics:
            self._cache[cache_key] = (topics, confidence)
            return topics, confidence

        # LLM fallback if rule-based classification is inconclusive
        prompt = (
            "Choose the single best topic for the statement from this taxonomy: "
            f"{', '.join(TOPIC_TAXONOMY)}.\n"
            'Return JSON only: {"topic": "...", "confidence": 0.0}\n'
            f"Statement: {statement}"
        )
        try:
            result = await self.llm.ainvoke(prompt, response_format="json", priority=LLMPriority.LOW)
            topic = (result.get("topic") or "other").strip().lower()
            conf = float(result.get("confidence", 0.0) or 0.0)
            if topic not in TOPIC_TAXONOMY:
                topic = "other"
                conf = 0.0
            topics = [] if topic == "other" else [topic]
            self._cache[cache_key] = (topics, conf)
            return topics, conf
        except Exception as e:
            logger.warning(f"[TopicClassifier] LLM fallback failed: {e}")
            self._cache[cache_key] = ([], 0.0)
            return [], 0.0


class DocTypeClassifier:
    @staticmethod
    def classify(source_url: str) -> str:
        url = (source_url or "").lower()
        if "pubmed" in url or "doi" in url or "journal" in url:
            return "journal"
        if "guideline" in url or "mmwr" in url:
            return "guideline"
        if "report" in url:
            return "report"
        if "news" in url or "press" in url:
            return "news"
        if "medlineplus.gov/ency" in url:
            return "encyclopedia"
        return "web"


class SourceMapper:
    _MAP = {
        "medlineplus.gov": "nlm",
        "nlm.nih.gov": "nlm",
        "pubmed.ncbi.nlm.nih.gov": "pubmed",
        "pmc.ncbi.nlm.nih.gov": "pubmed",
        "nih.gov": "nih",
        "cdc.gov": "cdc",
        "who.int": "who",
        "nhs.uk": "nhs",
    }

    @staticmethod
    def map(domain: str) -> str:
        if not domain:
            return "unknown"
        for key, val in SourceMapper._MAP.items():
            if domain.endswith(key):
                return val
        return domain


class FactTypeInferer:
    _COUNT_TOKENS = {
        "bones",
        "bone",
        "beats",
        "beat",
        "cells",
        "cell",
        "cases",
        "case",
        "times",
        "bpm",
    }

    @staticmethod
    def infer(statement: str) -> Tuple[str, float | None]:
        text = (statement or "").lower()
        nums = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)
        has_range = bool(re.search(r"\b\d+(\.\d+)?\s*(?:-|to)\s*\d+(\.\d+)?\b", text))

        if has_range:
            return "range", None

        if nums and any(tok in text for tok in FactTypeInferer._COUNT_TOKENS):
            value = nums[0].replace(",", "")
            try:
                return "count", float(value)
            except Exception:
                return "count", None

        if " is " in text or " are " in text or " was " in text or " were " in text:
            return "definition", None

        if any(k in text for k in ("causes", "leads to", "increases", "reduces", "associated")):
            return "mechanism", None

        return "other", None


@dataclass
class MetadataEnricher:
    topic_classifier: TopicClassifier = field(default_factory=TopicClassifier)

    async def enrich_fact(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        source_url = fact.get("source_url") or fact.get("source") or ""
        domain = extract_domain(source_url) or ""

        if not fact.get("domain"):
            fact["domain"] = domain
        if not fact.get("source"):
            fact["source"] = SourceMapper.map(domain)
        if not fact.get("doc_type"):
            fact["doc_type"] = DocTypeClassifier.classify(source_url)

        statement = fact.get("statement", "") or ""
        entities = fact.get("entities", []) or []

        topics, topic_conf = await self.topic_classifier.classify(statement, entities, source_url)
        if topics:
            fact["topic"] = topics[0]
        else:
            fact["topic"] = fact.get("topic") or "other"
        fact["topic_confidence"] = float(topic_conf)

        if not fact.get("fact_type"):
            fact_type, count_value = FactTypeInferer.infer(statement)
            fact["fact_type"] = fact_type
            if count_value is not None:
                fact["count_value"] = count_value

        return fact

    async def enrich_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not facts:
            return facts
        enriched = []
        for fact in facts:
            enriched.append(await self.enrich_fact(fact))
        return enriched
