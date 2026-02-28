"""
Corrective Retrieval Pipeline: Main orchestrator.

QUOTA-OPTIMIZED ARCHITECTURE (One Query at a Time):
    1. Embedding: Embed claim and extract entities
    2. Retrieval: Check VDB + KG for existing evidence FIRST
    3. Ranking: If trust score >= threshold, DONE (no web search needed)
    4. Query Generation: Generate search queries (1 LLM call)
    5. Incremental Search: For each query (one at a time):
       a. Execute single search API call
       b. Scrape URLs from that query
       c. Extract facts (LLM call)
       d. Ingest to VDB + KG
       e. Re-rank with new evidence
       f. If threshold met → STOP (save remaining quota)
       g. If not → continue to next query
    6. Return results

This approach MAXIMIZES quota efficiency by:
    - Using cached evidence when available (no API calls)
    - Running ONE search query at a time
    - Processing each query's results completely before deciding to continue
    - Stopping immediately when threshold is reached
    - Never wasting API calls on unused queries
"""

import os
import re
import urllib.parse
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.constants.config import (
    PIPELINE_CONF_THRESHOLD,
    PIPELINE_MAX_ROUNDS,
    PIPELINE_MAX_SEARCH_QUERIES,
    PIPELINE_MAX_URLS_PER_QUERY,
    PIPELINE_MIN_NEW_URLS,
    PIPELINE_RETRIEVAL_TOP_K,
)
from app.core.config import settings
from app.core.logger import get_logger
from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.pipeline.extraction_phase import extract_all, scrape_pages
from app.services.corrective.pipeline.ingestion_phase import ingest_facts_and_triples
from app.services.corrective.pipeline.ranking_phase import rank_candidates
from app.services.corrective.pipeline.retrieval_phase import retrieve_candidates
from app.services.corrective.relation_extractor import RelationExtractor
from app.services.corrective.scraper import Scraper
from app.services.corrective.trusted_search import TrustedSearch
from app.services.domain import HealthScopeGate
from app.services.kg.kg_ingest import KGIngest
from app.services.kg.kg_retrieval import KGRetrieval
from app.services.kg.neo4j_client import Neo4jClient
from app.services.llms.hybrid_service import get_groq_job_metadata, reset_groq_counter
from app.services.logging.log_handler import LogManagerHandler
from app.services.pipeline_debug_report import PipelineDebugReporter
from app.services.ranking.evidence_snapshot import make_evidence_snapshot_id
from app.services.ranking.trust_ranker import EvidenceItem, TrustRankingModule
from app.services.retrieval.lexical_index import LexicalIndex
from app.services.retrieval.metadata_enricher import TopicClassifier
from app.services.vdb.vdb_ingest import VDBIngest
from app.services.vdb.vdb_retrieval import VDBRetrieval
from app.services.verdict.verdict_generator import VerdictGenerator
from app.shared.trust_config import get_trust_config

logger = get_logger(__name__)


class CorrectivePipeline:
    """
    Corrective Retrieval Pipeline with integrated hybrid ranking.

    Orchestrates:
        1. Query reformulation (TrustedSearch -> LLM)
        2. Trusted Search on whitelisted domains
        3. Page scraping and content extraction
        4. Fact/Entity/Relation extraction (LLM-powered)
        5. Ingestion to VDB (Pinecone) and KG (Neo4j)
        6. Retrieval: semantic (VDB) + structural (KG)
        7. Hybrid ranking combining 5 scoring signals
        8. Reinforcement loop for low-confidence results
    """

    MAX_ROUNDS = PIPELINE_MAX_ROUNDS
    CONF_THRESHOLD = PIPELINE_CONF_THRESHOLD
    MIN_NEW_URLS = PIPELINE_MIN_NEW_URLS
    MAX_SEARCH_QUERIES = PIPELINE_MAX_SEARCH_QUERIES
    MAX_URLS_PER_QUERY = PIPELINE_MAX_URLS_PER_QUERY

    def __init__(self) -> None:
        # Search and scraping
        self.search_agent = TrustedSearch()
        self.scraper = Scraper()

        # Extraction
        self.fact_extractor = FactExtractor()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()

        # Storage and retrieval
        self.vdb_ingest = VDBIngest()  # Pinecone
        self.vdb_retriever = VDBRetrieval()
        self.kg_client = Neo4jClient()
        self.kg_ingest = KGIngest()
        self.kg_retriever = KGRetrieval()
        self.lexical_index = LexicalIndex()
        self.topic_classifier = TopicClassifier()
        self.health_scope_gate = HealthScopeGate()

        # Verdict generation (RAG phase)
        self.verdict_generator = VerdictGenerator()

        # Trust ranking
        self.trust_ranker = TrustRankingModule()

        # Logging system
        self.log_manager = LogManagerHandler._log_manager

    @staticmethod
    def _confidence_mode() -> bool:
        raw = os.getenv("LUXIA_CONFIDENCE_MODE")
        if raw is None:
            return bool(getattr(settings, "LUXIA_CONFIDENCE_MODE", False))
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _trust_payload_adaptive(adaptive_trust: Dict[str, Any]) -> Dict[str, Any]:
        trust_post = float(adaptive_trust.get("trust_post", 0.0) or 0.0)
        is_sufficient = bool(adaptive_trust.get("is_sufficient", False))
        return {
            "trust_policy_mode": "adaptive",
            "trust_metric_name": "adaptive_trust_post",
            "trust_metric_value": trust_post,
            "trust_threshold": "adaptive",
            "trust_threshold_met": is_sufficient,
            "adaptive_is_sufficient": is_sufficient,
            "adaptive_trust_post": trust_post,
            "trust_post_adaptive": trust_post,
        }

    @staticmethod
    def _trust_payload_fixed(trust_post: float, threshold: float) -> Dict[str, Any]:
        trust_post_f = float(trust_post or 0.0)
        threshold_f = float(threshold or 0.0)
        return {
            "trust_policy_mode": "fixed",
            "trust_metric_name": "trust_post",
            "trust_metric_value": trust_post_f,
            "trust_threshold": threshold_f,
            "trust_threshold_met": trust_post_f >= threshold_f,
        }

    @staticmethod
    def _cache_fast_path_allowed(adaptive_trust: Dict[str, Any], verdict_result: Dict[str, Any]) -> bool:
        if not bool(adaptive_trust.get("is_sufficient", False)):
            return False
        verdict = str(verdict_result.get("verdict") or "").upper()
        if verdict == "UNVERIFIABLE":
            return False

        # Deterministic gate from adaptive trust metrics (not LLM phrasing).
        coverage = float(adaptive_trust.get("coverage", 0.0) or 0.0)
        num_subclaims = int(adaptive_trust.get("num_subclaims", 0) or 0)
        strong_covered = int(adaptive_trust.get("strong_covered", 0) or 0)
        if num_subclaims > 0:
            if strong_covered >= num_subclaims and coverage >= 0.99:
                return True
            if coverage < 0.99:
                return False

        analysis_counts = verdict_result.get("analysis_counts", {}) or {}
        if analysis_counts:
            try:
                admissible_ratio = float(analysis_counts.get("admissible_evidence_ratio", 0.0) or 0.0)
            except Exception:
                admissible_ratio = 0.0
            if admissible_ratio < 0.35:
                return False
            rel_dist = analysis_counts.get("relevance_distribution", {}) or {}
            supports = (
                int(rel_dist.get("SUPPORTS", 0) or 0)
                + int(rel_dist.get("PARTIALLY_SUPPORTS", 0) or 0)
                + int(rel_dist.get("VALID", 0) or 0)
                + int(rel_dist.get("PARTIALLY_VALID", 0) or 0)
            )
            contradicts = (
                int(rel_dist.get("CONTRADICTS", 0) or 0)
                + int(rel_dist.get("PARTIALLY_CONTRADICTS", 0) or 0)
                + int(rel_dist.get("INVALID", 0) or 0)
                + int(rel_dist.get("PARTIALLY_INVALID", 0) or 0)
            )
            if (supports + contradicts) == 0:
                return False

        claim_breakdown = verdict_result.get("claim_breakdown", []) or []
        if claim_breakdown:
            unresolved_reasons = {
                str(((seg.get("alignment_debug") or {}).get("reason") or "")).lower() for seg in claim_breakdown
            }
            if "no_relevant_evidence" in unresolved_reasons or "insufficient_admissible_evidence" in unresolved_reasons:
                return False

        if "required_segments_resolved" in verdict_result:
            return bool(verdict_result.get("required_segments_resolved", False))

        if not claim_breakdown:
            return False
        return all((seg.get("status") or "UNKNOWN").upper() != "UNKNOWN" for seg in claim_breakdown)

    @staticmethod
    def _stance_score_components(stance: str) -> tuple[float, float]:
        stance_raw = {"entails": 1.0, "neutral": 0.0, "contradicts": -1.0}.get((stance or "neutral").lower(), 0.0)
        stance_mapped = (stance_raw + 1.0) / 2.0
        return stance_raw, stance_mapped

    @staticmethod
    def _log_top_domains(round_id: str, ranked: List[Dict[str, Any]]) -> None:
        domain_counts: Dict[str, int] = {}
        for item in ranked[:10]:
            src = str(item.get("source_url") or item.get("source") or "").strip()
            if not src:
                continue
            parsed = urllib.parse.urlparse(src)
            domain = (parsed.netloc or "").lower().removeprefix("www.")
            if not domain:
                continue
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if domain_counts:
            top_domains = sorted(domain_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            logger.info("[CorrectivePipeline:%s] Top evidence domains: %s", round_id, top_domains)

    @staticmethod
    def _classify_claim_frame(claim: str) -> Dict[str, Any]:
        text = f" {str(claim or '').strip().lower()} "
        therapeutic = bool(re.search(r"\b(cure|cures|treat|treats|therapy|effective against)\b", text))
        strong = bool(re.search(r"\b(cure|cures|eradicate|eliminate)\b", text))
        return {
            "claim_type": "THERAPEUTIC_EFFICACY" if therapeutic else "GENERIC",
            "strength": "STRONG" if strong else "NORMAL",
            "is_strong_therapeutic": therapeutic and strong,
        }

    @staticmethod
    def _is_comparative_claim(claim: str) -> bool:
        text = f" {str(claim or '').strip().lower()} "
        return bool(
            re.search(
                r"\b(more|less|fewer|greater|higher|lower|sooner|faster|slower|longer|shorter)\b.{0,80}\bthan\b"
                r"|\bvs\b|\bversus\b|\bcompared to\b",
                text,
            )
        )

    @staticmethod
    def _count_comparative_evidence_hits(claim: str, ranked: List[Dict[str, Any]]) -> int:
        claim_terms = {w for w in re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", (claim or "").lower())}
        stop = {"more", "less", "than", "with", "from", "lack", "human", "humans", "can", "die"}
        claim_terms = {w for w in claim_terms if w not in stop}
        hits = 0
        for item in ranked[:10]:
            stmt = str(item.get("statement") or item.get("text") or "")
            if not stmt:
                continue
            low = stmt.lower()
            has_comparison = bool(
                re.search(
                    r"\b(than|compared to|versus|vs|sooner|faster|slower|longer|shorter|higher|lower)\b",
                    low,
                )
            )
            has_quant = bool(
                re.search(r"\b\d+(?:\.\d+)?\b|\b(days?|hours?|weeks?|months?|years?|mortality|survival)\b", low)
            )
            stmt_terms = set(re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", low))
            overlap = len(claim_terms & stmt_terms)
            if has_comparison and (has_quant or overlap >= 2):
                hits += 1
        return hits

    @staticmethod
    def _estimate_information_gain(
        adaptive_trust: Dict[str, Any],
        zero_extraction_rounds: int = 0,
        low_yield_rounds: int = 0,
    ) -> float:
        coverage = float(adaptive_trust.get("coverage", 0.0) or 0.0)
        diversity = float(adaptive_trust.get("diversity", 0.0) or 0.0)
        agreement = float(adaptive_trust.get("agreement", 0.0) or 0.0)
        trust_post = float(adaptive_trust.get("trust_post", 0.0) or 0.0)
        contradicted = int(adaptive_trust.get("contradicted_subclaims", 0) or 0)
        signal_gap = abs(0.5 - trust_post) * 2.0  # 0..1; near 0.5 means ambiguous evidence.
        gain = (
            (0.34 * (1.0 - coverage))
            + (0.22 * (1.0 - diversity))
            + (0.16 * (1.0 - agreement))
            + (0.20 * signal_gap)
            + (0.08 * (1.0 if contradicted > 0 else 0.0))
        )
        if zero_extraction_rounds >= 2:
            gain *= 0.70
        if low_yield_rounds >= 2:
            gain *= 0.80
        return max(0.0, min(1.0, gain))

    @staticmethod
    def _should_stop_confidence_mode(
        claim_frame: Dict[str, Any],
        adaptive_trust: Dict[str, Any],
        confidence_target_coverage: float,
        confidence_max_new_trusted_urls: int,
        new_trusted_urls_processed: int,
        zero_extraction_rounds: int = 0,
        low_yield_rounds: int = 0,
    ) -> tuple[bool, str]:
        if new_trusted_urls_processed >= confidence_max_new_trusted_urls:
            return True, (
                f"processed trusted URL cap reached "
                f"({new_trusted_urls_processed} >= {confidence_max_new_trusted_urls})"
            )

        coverage = float(adaptive_trust.get("coverage", 0.0) or 0.0)
        diversity = float(adaptive_trust.get("diversity", 0.0) or 0.0)
        strong_covered = int(adaptive_trust.get("strong_covered", 0) or 0)
        is_sufficient = bool(adaptive_trust.get("is_sufficient", False))
        contradicted = int(adaptive_trust.get("contradicted_subclaims", 0) or 0)
        trust_post = float(adaptive_trust.get("trust_post", 0.0) or 0.0)
        agreement = float(adaptive_trust.get("agreement", 0.0) or 0.0)
        num_subclaims = int(adaptive_trust.get("num_subclaims", 0) or 0)
        info_gain = CorrectivePipeline._estimate_information_gain(
            adaptive_trust,
            zero_extraction_rounds=zero_extraction_rounds,
            low_yield_rounds=low_yield_rounds,
        )

        if is_sufficient:
            return True, (
                f"adaptive_sufficient=True coverage={coverage:.2f} diversity={diversity:.2f} "
                f"strong_covered={strong_covered} gain={info_gain:.3f}"
            )

        # Contradiction-focused early stop for non-therapeutic claims:
        # if contradiction signal is stable and broad enough, additional search rounds
        # usually add latency without improving decision quality.
        if (
            not claim_frame.get("is_strong_therapeutic", False)
            and contradicted >= max(1, num_subclaims)
            and agreement >= 0.90
            and diversity >= 0.60
            and new_trusted_urls_processed >= 6
            and trust_post <= 0.20
        ):
            return True, (
                "stable contradiction stop "
                f"(contradicted={contradicted}, agreement={agreement:.2f}, "
                f"diversity={diversity:.2f}, trust_post={trust_post:.3f}, "
                f"processed={new_trusted_urls_processed}, gain={info_gain:.3f})"
            )

        if claim_frame.get("is_strong_therapeutic", False):
            # Strong therapeutic claims require stronger stopping signals than coverage-only.
            if contradicted > 0 and strong_covered >= 1:
                return True, (
                    f"strong therapeutic contradiction found (contradicted={contradicted}, "
                    f"strong_covered={strong_covered}, gain={info_gain:.3f})"
                )
            if coverage >= confidence_target_coverage and strong_covered >= 1:
                return True, (
                    f"strong therapeutic coverage target met with strong evidence "
                    f"(coverage={coverage:.2f}, strong_covered={strong_covered}, gain={info_gain:.3f})"
                )
            return False, (
                f"continue: strong therapeutic claim requires strong efficacy evidence "
                f"(coverage={coverage:.2f}, diversity={diversity:.2f}, strong_covered={strong_covered}, "
                f"sufficient={is_sufficient}, gain={info_gain:.3f})"
            )

        if zero_extraction_rounds >= 2 and info_gain < 0.15:
            return True, (
                f"low-yield saturation stop (zero_extraction_rounds={zero_extraction_rounds}, "
                f"low_yield_rounds={low_yield_rounds}, gain={info_gain:.3f})"
            )

        if low_yield_rounds >= 3 and new_trusted_urls_processed >= 6 and info_gain < 0.18:
            return True, (
                f"expected-gain stop (low_yield_rounds={low_yield_rounds}, processed={new_trusted_urls_processed}, "
                f"gain={info_gain:.3f})"
            )

        if (
            is_sufficient
            and coverage >= confidence_target_coverage
            and diversity >= 0.45
            and (trust_post >= 0.55 or strong_covered >= 1)
        ):
            return True, (
                f"coverage target reached with diversity guard "
                f"(coverage={coverage:.2f} >= {confidence_target_coverage:.2f}, diversity={diversity:.2f}, "
                f"trust_post={trust_post:.3f}, strong_covered={strong_covered}, gain={info_gain:.3f})"
            )

        if coverage >= confidence_target_coverage and not is_sufficient:
            return False, (
                f"continue: coverage reached but adaptive_sufficient=False "
                f"(coverage={coverage:.2f}, diversity={diversity:.2f}, gain={info_gain:.3f})"
            )

        return False, (
            f"continue: coverage/diversity guard not met "
            f"(coverage={coverage:.2f}, diversity={diversity:.2f}, sufficient={is_sufficient}, gain={info_gain:.3f})"
        )

    def _build_evidence_items(self, claim: str, ranked: List[Dict[str, Any]]) -> List[EvidenceItem]:
        evidence_items = [
            EvidenceItem(
                statement=item.get("statement", item.get("fact", "")),
                semantic_score=item.get("sem_score", item.get("final_score", item.get("score", 0.0))),
                source_url=item.get("source_url", item.get("source", "")),
                published_at=item.get("published_at", item.get("publish_date")),
                trust=item.get("final_score", item.get("score", 0.0)),
                stance=str(item.get("stance", "neutral") or "neutral"),
                score_components={
                    "semantic": item.get("sem_score", item.get("final_score", item.get("score", 0.0))),
                    "source": 0.5,
                    "recency": 0.5,
                    "stance_raw": 0.0,
                    "stance_mapped": 0.5,
                    "trust": item.get("final_score", item.get("score", 0.0)),
                    "claim_overlap": float(item.get("claim_overlap", 0.0) or 0.0),
                    "anchor_match_score": float(item.get("anchor_match_score", 0.0) or 0.0),
                    "predicate_match_score": float(item.get("predicate_match_score", 0.0) or 0.0),
                    "object_relation_overlap": float(item.get("object_relation_overlap", 0.0) or 0.0),
                    "topic_aligned": bool(item.get("topic_aligned", True)),
                    "admissible_for_trust": bool(item.get("admissible_for_trust", True)),
                },
                candidate_type=str(item.get("candidate_type", "") or ""),
            )
            for item in ranked
        ]
        try:
            self.trust_ranker.classify_stance_for_evidence(claim, evidence_items)
        except Exception as e:
            logger.warning("[CorrectivePipeline] stance classification failed; continuing with neutral stance: %s", e)
            return evidence_items

        for ev in evidence_items:
            stance_raw, stance_mapped = self._stance_score_components(ev.stance)
            components = ev.score_components or {}
            components["stance_raw"] = stance_raw
            components["stance_mapped"] = stance_mapped
            ev.score_components = components
        return evidence_items

    @staticmethod
    def _evidence_snapshot_id_from_ranked(claim: str, ranked: List[Dict[str, Any]]) -> str:
        class _SnapshotEv:
            __slots__ = ("id", "domain", "source_url", "text")

            def __init__(self, idx: int, item: Dict[str, Any]) -> None:
                self.id = idx
                src = str(item.get("source_url") or item.get("source") or "")
                self.source_url = src
                self.domain = urllib.parse.urlparse(src).netloc.lower().removeprefix("www.")
                self.text = str(item.get("statement") or item.get("text") or "")

        items = [_SnapshotEv(i, item) for i, item in enumerate(ranked or [])]
        return make_evidence_snapshot_id(items, salt=claim).snapshot_id

    async def run(
        self,
        post_text: str,
        domain: str,
        failed_entities: Optional[List[str]] = None,
        round_id: Optional[str] = None,
        top_k: int = 5,
        stage_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None] | None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the OPTIMIZED corrective retrieval pipeline.

        RETRIEVAL-FIRST APPROACH:
        1. Extract entities from claim (single LLM call)
        2. Retrieve existing evidence from VDB + KG
        3. Rank existing evidence
        4. IF trust score >= threshold: DONE (no web search, no extra LLM calls!)
        5. ELSE: Search incrementally, process in batches until threshold reached

        This minimizes Groq API calls by using cached knowledge when available.
        """
        round_id = round_id or str(uuid.uuid4())
        failed_entities = failed_entities or []
        confidence_mode = self._confidence_mode()
        try:
            min_internal_top_k = max(3, int(os.getenv("PIPELINE_MIN_INTERNAL_TOP_K", "3")))
        except Exception:
            min_internal_top_k = 3
        if confidence_mode:
            min_internal_top_k = max(min_internal_top_k, int(os.getenv("CONFIDENCE_TOP_K", "10")))
        internal_top_k = max(int(top_k or 1), min_internal_top_k)

        # Reset Groq call counter for this new request (per-job)
        reset_groq_counter(job_id=round_id)
        self.scraper.reset_job_attempts()
        debug_reporter = PipelineDebugReporter(run_id=round_id)
        await debug_reporter.initialize(post_text)

        logger.info(f"[CorrectivePipeline:{round_id}] Start for post: {post_text}")
        normalized_domain = "health"
        if str(domain or "").strip().lower() not in {"", "health", "biomedical", "medicine"}:
            logger.warning(
                "[CorrectivePipeline:%s] Domain '%s' requested; enforcing health-only runtime domain.",
                round_id,
                domain,
            )
        health_scope = self.health_scope_gate.classify(post_text, declared_domain=domain)

        async def _emit_stage(stage: str, payload: Dict[str, Any] | None = None) -> None:
            if not stage_callback:
                return
            body = {"round_id": round_id, **(payload or {})}
            maybe = stage_callback(stage, body)
            if maybe is not None:
                await maybe

        await _emit_stage(
            "started",
            {
                "claim_preview": post_text[:120],
                "domain": normalized_domain,
                "health_scope": {
                    "in_scope": health_scope.health_in_scope,
                    "biomedical_confidence": health_scope.biomedical_confidence,
                    "scope_reason": health_scope.scope_reason,
                },
            },
        )

        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Pipeline started (retrieval-first): {post_text[:100]}...",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={
                    "domain": normalized_domain,
                    "health_scope": {
                        "in_scope": health_scope.health_in_scope,
                        "biomedical_confidence": health_scope.biomedical_confidence,
                        "scope_reason": health_scope.scope_reason,
                    },
                },
            )

        if not health_scope.health_in_scope:
            verdict_result = {
                "verdict": "UNVERIFIABLE",
                "display_verdict": "UNVERIFIABLE",
                "verdict_band": "MIXED_OR_UNCLEAR",
                "confidence": 0.22,
                "truthfulness_percent": 35.0,
                "truth_score_percent": 35.0,
                "rationale": (
                    "The input does not provide enough biomedical signal for health-domain evidence adjudication. "
                    "Returning UNVERIFIABLE conservatively."
                ),
                "claim_breakdown": [
                    {
                        "claim_segment": post_text,
                        "status": "UNKNOWN",
                        "supporting_fact": "",
                        "source_url": "",
                        "evidence_used_ids": [],
                    }
                ],
                "evidence_map": [],
                "key_findings": [],
                "class_probs": {
                    "true": 0.20,
                    "false": 0.20,
                    "unverifiable": 0.60,
                },
                "calibrated_confidence": 0.22,
                "calibration_meta": {
                    "mode": "health_scope_gate",
                    "scope_reason": health_scope.scope_reason,
                    "biomedical_confidence": health_scope.biomedical_confidence,
                },
                "trust_threshold_met": False,
            }
            await _emit_stage(
                "completed",
                {
                    "status": "out_of_scope",
                    "health_scope": {
                        "in_scope": health_scope.health_in_scope,
                        "biomedical_confidence": health_scope.biomedical_confidence,
                        "scope_reason": health_scope.scope_reason,
                    },
                },
            )
            await debug_reporter.log_step(
                step_name="Pipeline completed",
                description="Exited early on health-domain scope gate",
                input_data={"claim": post_text},
                output_data={"status": "out_of_scope", "verdict": verdict_result["verdict"]},
            )
            await debug_reporter.close()
            return {
                "round_id": round_id,
                "status": "out_of_scope",
                "facts": [],
                "triples": [],
                "queries": [],
                "queries_total": 0,
                "semantic_candidates_count": 0,
                "kg_candidates_count": 0,
                "ranked": [],
                "used_web_search": False,
                "data_source": "DOMAIN_GATE",
                "search_api_calls": 0,
                "search_api_calls_saved": 0,
                "urls_processed": 0,
                "urls_skipped_already_processed": 0,
                "initial_top_score": 0.0,
                "ranking_top_score": 0.0,
                "ranking_avg_score": 0.0,
                "trust_post": 0.0,
                "trust_post_adaptive": 0.0,
                "trust_post_fixed": 0.0,
                "trust_grade": "F",
                "agreement_ratio": 0.0,
                "coverage": 0.0,
                "diversity": 0.0,
                "num_subclaims": 1,
                "adaptive_is_sufficient": False,
                "fixed_trust_threshold": self.CONF_THRESHOLD,
                "fixed_trust_threshold_met": False,
                "verdict": verdict_result,
                "debug_counts": {
                    "kg_raw": 0,
                    "kg_with_score": 0,
                    "kg_in_ranked": 0,
                    "sem_raw": 0,
                    "sem_filtered": 0,
                    "sem_in_ranked": 0,
                },
                "vdb_signal_count": 0,
                "kg_signal_count": 0,
                "vdb_signal_sum_top5": 0.0,
                "kg_signal_sum_top5": 0.0,
                "llm": get_groq_job_metadata(),
                "trust_policy_mode": "adaptive",
                "trust_metric_name": "adaptive_trust_post",
                "trust_metric_value": 0.0,
                "trust_threshold": "adaptive",
                "trust_threshold_met": False,
                "domain": normalized_domain,
                "health_scope": {
                    "in_scope": health_scope.health_in_scope,
                    "biomedical_confidence": health_scope.biomedical_confidence,
                    "scope_reason": health_scope.scope_reason,
                },
                "pipeline_diagnostics_v2": {
                    "stop_reason": "health_scope_gate",
                    "gain_estimate": 0.0,
                    "kg_timeout_count": 0,
                    "zero_extraction_rounds": 0,
                },
            }

        # ====================================================================
        # PHASE 1: Extract entities from claim (1 LLM call)
        # ====================================================================
        claim_entities = await self._extract_claim_entities(post_text, round_id)
        must_have_entities = self._select_must_have_entities(post_text, claim_entities)
        must_have_aliases = self._build_must_have_aliases(claim_entities, must_have_entities)
        await debug_reporter.log_step(
            step_name="Claim entities identified",
            description="Entity extraction on incoming claim",
            input_data={"claim": post_text},
            output_data={
                "claim_entities": claim_entities,
                "must_have_entities": must_have_entities,
                "must_have_aliases": must_have_aliases,
            },
        )
        logger.info(
            f"[CorrectivePipeline:{round_id}] Claim entities: {claim_entities} | "
            f"must_have={must_have_entities} aliases={must_have_aliases}"
        )

        # ====================================================================
        # PHASE 2: Retrieve existing evidence from VDB + KG (NO LLM calls)
        # ====================================================================
        claim_topics, topic_conf = await self._infer_claim_topics(post_text, claim_entities)
        logger.info(f"[CorrectivePipeline:{round_id}] Claim topics: {claim_topics} (conf={topic_conf:.2f})")

        retrieval_queries = self._build_retrieval_queries(post_text)
        await debug_reporter.log_step(
            step_name="Subclaims identified",
            description="Claim decomposition and retrieval query seed construction",
            input_data={"claim": post_text},
            output_data={"retrieval_queries": retrieval_queries},
        )
        raw_retrieval_top_k = max(PIPELINE_RETRIEVAL_TOP_K, top_k * 3)
        dedup_sem, kg_candidates, retrieval_metrics = await retrieve_candidates(
            self.vdb_retriever,
            self.kg_retriever,
            retrieval_queries,
            claim_entities,
            raw_retrieval_top_k,
            round_id,
            claim_topics,
            self.lexical_index,
            self.log_manager,
            post_text,
            claim_anchors=must_have_aliases or claim_entities,
            include_metrics=True,
        )
        debug_counts = {
            "kg_raw": int(retrieval_metrics.get("kg_raw", len(kg_candidates))),
            "kg_with_score": int(retrieval_metrics.get("kg_with_score", 0)),
            "kg_in_ranked": 0,
            "sem_raw": int(retrieval_metrics.get("sem_raw", 0)),
            "sem_filtered": int(retrieval_metrics.get("sem_filtered", len(dedup_sem))),
            "sem_in_ranked": 0,
        }
        await _emit_stage(
            "retrieval_done",
            {
                "semantic_candidates_count": len(dedup_sem),
                "kg_candidates_count": len(kg_candidates),
                "debug_counts": debug_counts,
            },
        )

        logger.info(
            f"[CorrectivePipeline:{round_id}] Retrieved {len(dedup_sem)} VDB + {len(kg_candidates)} KG candidates"
        )
        await debug_reporter.log_step(
            step_name="Outputs of each VB/KG retrieval",
            description="Raw retrieval output before ranking",
            input_data={"retrieval_queries": retrieval_queries, "claim_entities": claim_entities},
            output_data={
                "semantic_candidates_count": len(dedup_sem),
                "kg_candidates_count": len(kg_candidates),
                "semantic_candidates": dedup_sem,
                "kg_candidates": kg_candidates,
            },
        )

        # ====================================================================
        # PHASE 3: Rank existing evidence (NO LLM calls)
        # ====================================================================
        top_ranked = []
        if dedup_sem or kg_candidates:
            top_ranked = await rank_candidates(
                dedup_sem,
                kg_candidates,
                claim_entities,
                post_text,
                internal_top_k,
                round_id,
                self.log_manager,
                must_have_entities=must_have_aliases or claim_entities[:1],
            )
        debug_counts["kg_in_ranked"] = sum(1 for r in top_ranked if float(r.get("kg_score", 0.0) or 0.0) > 0.0)
        debug_counts["sem_in_ranked"] = sum(1 for r in top_ranked if float(r.get("sem_score", 0.0) or 0.0) > 0.0)
        await _emit_stage(
            "ranking_done",
            {
                "ranked_count": len(top_ranked),
                "kg_in_ranked": debug_counts["kg_in_ranked"],
                "sem_in_ranked": debug_counts["sem_in_ranked"],
            },
        )
        self._log_top_domains(round_id, top_ranked)

        # Compute trust scores for evidence

        top_ranked_evidence = self._build_evidence_items(post_text, top_ranked)

        # Use adaptive trust policy for multi-part claims
        adaptive_trust = self.trust_ranker.compute_adaptive_post_trust(post_text, top_ranked_evidence, internal_top_k)
        initial_fixed_post = self.trust_ranker.compute_post_trust(top_ranked_evidence, internal_top_k)
        initial_trust_snapshot_v2 = self.trust_ranker.compute_uncertainty_snapshot(top_ranked_evidence, internal_top_k)
        initial_trust_snapshot_v2["sufficiency_reason"] = str(adaptive_trust.get("gate_reason") or "")
        initial_trust_snapshot_v2["info_gain"] = 0.0
        initial_fixed_trust_score = float(initial_fixed_post.get("trust_post", 0.0) or 0.0)
        initial_fixed_payload = self._trust_payload_fixed(initial_fixed_trust_score, self.CONF_THRESHOLD)
        is_sufficient = adaptive_trust["is_sufficient"]
        adaptive_payload = self._trust_payload_adaptive(adaptive_trust)
        adaptive_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"[CorrectivePipeline:{round_id}] Adaptive trust: sufficient={is_sufficient}, "
            f"coverage={adaptive_trust['coverage']:.2f}, diversity={adaptive_trust['diversity']:.2f}"
        )

        current_snapshot_id = self._evidence_snapshot_id_from_ranked(post_text, top_ranked)
        adaptive_cache[current_snapshot_id] = adaptive_trust
        if is_sufficient:
            logger.info(
                f"[CorrectivePipeline:{round_id}] Adaptive trust threshold met. "
                "Running cache verdict precheck before deciding web search."
            )

            verdict_result = await self.verdict_generator.generate_verdict(
                claim=post_text,
                ranked_evidence=top_ranked,
                top_k=internal_top_k,
                used_web_search=False,
                cache_sufficient=True,
                adaptive_metrics=adaptive_trust,
                evidence_snapshot_id=current_snapshot_id,
            )

            logger.info(
                f"[CorrectivePipeline:{round_id}] Verdict (cache-sufficient): {verdict_result['verdict']} "
                f"(confidence: {verdict_result['confidence']:.2f})"
            )
            cache_allowed = self._cache_fast_path_allowed(adaptive_trust, verdict_result)
            if cache_allowed:
                await _emit_stage("search_done", {"search_api_calls": 0, "queries_total": 0})
                await _emit_stage("extraction_done", {"facts": 0, "triples": 0})
                await _emit_stage("ingestion_done", {"facts_ingested": 0, "triples_ingested": 0})
                await _emit_stage("completed", {"status": "completed_from_cache"})
                await debug_reporter.log_step(
                    step_name="Pipeline completed from cache",
                    description="Final output emitted without web search",
                    input_data={"used_web_search": False},
                    output_data={"ranked_count": len(top_ranked), "status": "completed_from_cache"},
                )
                await debug_reporter.close()
                return {
                    "round_id": round_id,
                    "status": "completed_from_cache",
                    "facts": [],  # No new facts extracted
                    "triples": [],
                    "queries": [post_text],
                    "semantic_candidates_count": len(dedup_sem),
                    "kg_candidates_count": len(kg_candidates),
                    "ranked": top_ranked,
                    "used_web_search": False,
                    "data_source": "CACHE",
                    "initial_top_score": adaptive_trust["trust_post"],
                    "ranking_top_score": (top_ranked[0].get("final_score", 0.0) if top_ranked else 0.0),
                    "ranking_avg_score": (
                        sum(float(r.get("final_score", 0.0) or 0.0) for r in top_ranked[:5])
                        / max(1, len(top_ranked[:5]))
                        if top_ranked
                        else 0.0
                    ),
                    "trust_post": adaptive_trust["trust_post"],
                    "trust_post_adaptive": adaptive_trust["trust_post"],
                    "trust_post_fixed": initial_fixed_trust_score,
                    "trust_snapshot_v2": initial_trust_snapshot_v2,
                    "trust_grade": "adaptive",  # Adaptive grading
                    "agreement_ratio": adaptive_trust["agreement"],
                    "coverage": adaptive_trust["coverage"],
                    "diversity": adaptive_trust["diversity"],
                    "num_subclaims": adaptive_trust["num_subclaims"],
                    "adaptive_is_sufficient": bool(adaptive_trust.get("is_sufficient", False)),
                    "fixed_trust_threshold": self.CONF_THRESHOLD,
                    "fixed_trust_threshold_met": bool(initial_fixed_payload.get("trust_threshold_met", False)),
                    "verdict": verdict_result,
                    "cache_sufficient": True,
                    "debug_counts": debug_counts,
                    "vdb_signal_count": debug_counts["sem_in_ranked"],
                    "kg_signal_count": debug_counts["kg_in_ranked"],
                    "vdb_signal_sum_top5": round(
                        sum(float(e.get("sem_score", 0.0) or 0.0) for e in top_ranked[:5]),
                        3,
                    ),
                    "kg_signal_sum_top5": round(
                        sum(float(e.get("kg_score", 0.0) or 0.0) for e in top_ranked[:5]),
                        3,
                    ),
                    "llm": get_groq_job_metadata(),
                    "domain": normalized_domain,
                    "health_scope": {
                        "in_scope": health_scope.health_in_scope,
                        "biomedical_confidence": health_scope.biomedical_confidence,
                        "scope_reason": health_scope.scope_reason,
                    },
                    "pipeline_diagnostics_v2": {
                        "stop_reason": "cache_sufficient",
                        "gain_estimate": 0.0,
                        "kg_timeout_count": 0,
                        "zero_extraction_rounds": 0,
                    },
                    **adaptive_payload,
                }
            logger.info(
                f"[CorrectivePipeline:{round_id}] Cache precheck unresolved required segments; "
                "marking adaptive_sufficient=False and continuing to web search."
            )
            adaptive_trust = {
                **adaptive_trust,
                "is_sufficient": False,
                "gate_reason": "cache_precheck_unresolved",
            }
            adaptive_payload = self._trust_payload_adaptive(adaptive_trust)
            adaptive_cache[current_snapshot_id] = adaptive_trust

        # ====================================================================
        # PHASE 4: Quota-Optimized Incremental Search (ONE QUERY AT A TIME)
        # ====================================================================
        logger.info(
            f"[CorrectivePipeline:{round_id}] Evidence insufficient (coverage={adaptive_trust['coverage']:.2f}, "
            f"diversity={adaptive_trust['diversity']:.2f}), starting quota-optimized search..."
        )
        if confidence_mode:
            logger.info("[CorrectivePipeline:%s] CONFIDENCE_MODE active", round_id)

        # Generate all search queries upfront (1 LLM call only)
        raw_subclaims = self.trust_ranker.adaptive_policy.decompose_claim(post_text)
        query_subclaims = [s.strip() for s in raw_subclaims if s and s.strip()]
        queries = await self.search_agent.generate_search_queries(
            post_text,
            failed_entities,
            max_queries=self.MAX_SEARCH_QUERIES,
            subclaims=query_subclaims,
            entities=claim_entities,
        )
        await debug_reporter.log_step(
            step_name="Google search queries generated",
            description="Query generation before web search execution",
            input_data={"raw_subclaims": raw_subclaims, "entities": claim_entities},
            output_data={"queries": queries, "query_count": len(queries)},
        )

        if not queries:
            logger.warning(f"[CorrectivePipeline:{round_id}] No search queries generated")

            # Generate verdict with whatever evidence we have
            verdict_result = await self.verdict_generator.generate_verdict(
                claim=post_text,
                ranked_evidence=top_ranked,
                top_k=internal_top_k,
                used_web_search=False,
            )
            await _emit_stage("search_done", {"search_api_calls": 0, "queries_total": 0})
            await _emit_stage("extraction_done", {"facts": 0, "triples": 0})
            await _emit_stage("ingestion_done", {"facts_ingested": 0, "triples_ingested": 0})
            await _emit_stage("completed", {"status": "no_queries_generated"})
            await debug_reporter.log_step(
                step_name="Pipeline halted: no queries generated",
                description="No search queries available after decomposition/reformulation",
                input_data={"claim": post_text},
                output_data={"status": "no_queries_generated", "ranked_count": len(top_ranked)},
            )
            await debug_reporter.close()

            return {
                "round_id": round_id,
                "status": "no_queries_generated",
                "facts": [],
                "triples": [],
                "queries": [],
                "semantic_candidates_count": len(dedup_sem),
                "kg_candidates_count": len(kg_candidates),
                "ranked": top_ranked,
                "used_web_search": False,
                "data_source": "CACHE",
                "search_api_calls": 0,
                "initial_top_score": adaptive_trust["trust_post"],
                "ranking_top_score": (top_ranked[0].get("final_score", 0.0) if top_ranked else 0.0),
                "ranking_avg_score": (
                    sum(float(r.get("final_score", 0.0) or 0.0) for r in top_ranked[:5]) / max(1, len(top_ranked[:5]))
                    if top_ranked
                    else 0.0
                ),
                "trust_post": adaptive_trust["trust_post"],
                "trust_post_adaptive": adaptive_trust["trust_post"],
                "trust_post_fixed": initial_fixed_trust_score,
                "trust_snapshot_v2": initial_trust_snapshot_v2,
                "coverage": adaptive_trust["coverage"],
                "diversity": adaptive_trust["diversity"],
                "num_subclaims": adaptive_trust["num_subclaims"],
                "adaptive_is_sufficient": bool(adaptive_trust.get("is_sufficient", False)),
                "fixed_trust_threshold": self.CONF_THRESHOLD,
                "fixed_trust_threshold_met": bool(initial_fixed_payload.get("trust_threshold_met", False)),
                "verdict": verdict_result,
                "debug_counts": debug_counts,
                "vdb_signal_count": debug_counts["sem_in_ranked"],
                "kg_signal_count": debug_counts["kg_in_ranked"],
                "vdb_signal_sum_top5": round(
                    sum(float(e.get("sem_score", 0.0) or 0.0) for e in top_ranked[:5]),
                    3,
                ),
                "kg_signal_sum_top5": round(
                    sum(float(e.get("kg_score", 0.0) or 0.0) for e in top_ranked[:5]),
                    3,
                ),
                "domain": normalized_domain,
                "health_scope": {
                    "in_scope": health_scope.health_in_scope,
                    "biomedical_confidence": health_scope.biomedical_confidence,
                    "scope_reason": health_scope.scope_reason,
                },
                "pipeline_diagnostics_v2": {
                    "stop_reason": "no_queries_generated",
                    "gain_estimate": 0.0,
                    "kg_timeout_count": 0,
                    "zero_extraction_rounds": 0,
                },
                "llm": get_groq_job_metadata(),
                **adaptive_payload,
            }

        # Get already-processed URLs from VDB to avoid re-scraping.
        # Scope by inferred topics and cap size to avoid stale/shared namespace contamination.
        try:
            max_vdb_url_skip = max(5, int(os.getenv("PIPELINE_MAX_VDB_URL_SKIP", "30")))
        except Exception:
            max_vdb_url_skip = 30
        already_processed_urls = self.vdb_ingest.get_processed_urls(
            topics=claim_topics,
            max_urls=max_vdb_url_skip + 1,
        )
        if len(already_processed_urls) > max_vdb_url_skip:
            logger.warning(
                "[CorrectivePipeline:%s] Processed URL set exceeded cap (%d > %d); "
                "truncating URL-skip filter to cap to reduce stale/shared namespace leakage",
                round_id,
                len(already_processed_urls),
                max_vdb_url_skip,
            )
            already_processed_urls = set(sorted(already_processed_urls)[:max_vdb_url_skip])
        logger.info(
            f"[CorrectivePipeline:{round_id}] Found {len(already_processed_urls)} " "already-processed URLs in VDB"
        )

        # Process ONE QUERY AT A TIME for maximum quota efficiency
        all_facts: List[Dict[str, Any]] = []
        all_triples: List[Dict[str, Any]] = []
        all_entities = list(claim_entities)
        processed_urls: List[str] = []
        skipped_urls: List[str] = []
        search_api_calls = 0
        queries_executed: List[str] = []
        zero_extraction_rounds = 0
        low_yield_rounds = 0
        kg_timeout_count = 0
        stop_reason = "query_budget_exhausted"
        gain_estimate = 0.0
        confidence_target_coverage = float(os.getenv("CONFIDENCE_TARGET_COVERAGE", "0.5"))
        confidence_max_new_trusted_urls = max(
            1,
            int(
                os.getenv(
                    "CONFIDENCE_MAX_NEW_TRUSTED_URLS",
                    str(get_trust_config().search_max_urls_confidence_mode),
                )
            ),
        )
        claim_frame = self._classify_claim_frame(post_text)
        logger.info(
            "[CorrectivePipeline:%s] Claim frame: type=%s strength=%s strong_therapeutic=%s",
            round_id,
            claim_frame["claim_type"],
            claim_frame["strength"],
            claim_frame["is_strong_therapeutic"],
        )
        is_comparative_claim = self._is_comparative_claim(post_text)
        comparative_monopoly_rounds = 0
        best_comparative_hits = 0
        new_trusted_urls_processed = 0
        for query_idx, query in enumerate(queries):
            # OPTIMIZATION: Hard limit on search queries to prevent runaway searches
            if search_api_calls >= self.MAX_SEARCH_QUERIES:
                logger.info(
                    f"[CorrectivePipeline:{round_id}] Reached MAX_SEARCH_QUERIES ({self.MAX_SEARCH_QUERIES}), "
                    f"stopping to conserve quota. {len(queries) - query_idx} queries unused."
                )
                stop_reason = "max_search_queries_reached"
                break

            logger.info(f"[CorrectivePipeline:{round_id}] === Query {query_idx + 1}/{len(queries)} ===")
            logger.info(f"[CorrectivePipeline:{round_id}] Executing: '{query}'")

            # Step 1: Execute SINGLE search API call
            search_api_calls += 1
            if query_idx == 0 and not confidence_mode:
                query_urls = await self.search_agent.search_for_claim(
                    post_text,
                    min_urls=1,
                    max_queries=1,
                )
            else:
                query_urls = await self.search_agent.execute_single_query(
                    query,
                    claim=post_text,
                    entities=claim_entities,
                )
            queries_executed.append(query)
            await debug_reporter.log_step(
                step_name="Number of URLs per search query",
                description="Search output for one query",
                input_data={"query_index": query_idx + 1, "query": query},
                output_data={"url_count": len(query_urls), "urls": query_urls},
            )
            await _emit_stage(
                "search_done",
                {
                    "query_index": query_idx + 1,
                    "queries_executed": len(queries_executed),
                    "search_api_calls": search_api_calls,
                    "urls_found": len(query_urls),
                },
            )

            if not query_urls:
                logger.info(
                    f"[CorrectivePipeline:{round_id}] Query {query_idx + 1} returned no URLs, " "trying next query..."
                )
                continue

            logger.info(f"[CorrectivePipeline:{round_id}] Query {query_idx + 1} returned " f"{len(query_urls)} URLs")

            # Filter out already-processed URLs to avoid re-scraping
            new_urls = [url for url in query_urls if url not in already_processed_urls and url not in processed_urls]
            skipped_this_query = len(query_urls) - len(new_urls)

            if skipped_this_query > 0:
                skipped_urls.extend([url for url in query_urls if url in already_processed_urls])
                logger.info(
                    f"[CorrectivePipeline:{round_id}] Filtered {skipped_this_query} already-processed URLs, "
                    f"{len(new_urls)} new URLs to scrape"
                )

            if not new_urls:
                logger.info(
                    f"[CorrectivePipeline:{round_id}] All URLs from query {query_idx + 1} already processed, "
                    "trying next query..."
                )
                continue

            # Step 2: Scrape only NEW URLs from this query
            if len(new_urls) > self.MAX_URLS_PER_QUERY:
                logger.info(
                    f"[CorrectivePipeline:{round_id}] Capping URLs to {self.MAX_URLS_PER_QUERY} "
                    f"from {len(new_urls)} candidates for this query"
                )
                new_urls = new_urls[: self.MAX_URLS_PER_QUERY]
            new_trusted_urls_processed += len(new_urls)
            scraped_pages = await scrape_pages(
                self.scraper,
                new_urls,
                round_id,
                self.log_manager,
                debug_reporter=debug_reporter,
            )
            processed_urls.extend(new_urls)
            # Add to already_processed set to avoid re-scraping in subsequent queries
            already_processed_urls.update(new_urls)

            if not scraped_pages:
                logger.info(f"[CorrectivePipeline:{round_id}] No content scraped from query {query_idx + 1}")
                low_yield_rounds += 1
                continue

            # Step 3: Extract facts, entities, relations
            query_facts, query_entities, query_triples = await extract_all(
                self.fact_extractor,
                self.entity_extractor,
                self.relation_extractor,
                scraped_pages,
                round_id,
                self.log_manager,
                debug_reporter=debug_reporter,
                claim_text=post_text,
                claim_entities=claim_entities,
                must_have_entities=must_have_aliases or claim_entities[:1],
            )
            await _emit_stage(
                "extraction_done",
                {
                    "query_index": query_idx + 1,
                    "facts": len(query_facts),
                    "entities": len(query_entities),
                    "triples": len(query_triples),
                },
            )

            if not query_facts:
                logger.info(f"[CorrectivePipeline:{round_id}] No facts extracted from query {query_idx + 1}")
                zero_extraction_rounds += 1
                low_yield_rounds += 1
                continue
            zero_extraction_rounds = 0
            if len(query_facts) <= 1:
                low_yield_rounds += 1
            else:
                low_yield_rounds = 0

            # Accumulate results
            all_facts.extend(query_facts)
            all_entities.extend(query_entities)
            all_triples.extend(query_triples)

            logger.info(
                f"[CorrectivePipeline:{round_id}] Query {query_idx + 1} extracted "
                f"{len(query_facts)} facts, {len(query_triples)} triples"
            )

            # Step 4: Ingest to VDB and KG immediately
            ingest_diag = await ingest_facts_and_triples(
                self.vdb_ingest,
                self.kg_ingest,
                query_facts,
                query_triples,
                round_id,
                self.log_manager,
            )
            kg_timeout_count += int(ingest_diag.get("kg_timeout_count", 0) or 0)
            await _emit_stage(
                "ingestion_done",
                {
                    "query_index": query_idx + 1,
                    "facts_ingested": len(query_facts),
                    "triples_ingested": len(query_triples),
                    "kg_timeout_count": kg_timeout_count,
                },
            )

            # Step 5: Re-retrieve and re-rank with new evidence
            # Keep retrieval anchored to claim entities to avoid topic drift from scraped-page artifacts.
            retrieval_entities = list(set(claim_entities + failed_entities))
            dedup_sem, kg_candidates, retrieval_metrics = await retrieve_candidates(
                self.vdb_retriever,
                self.kg_retriever,
                queries_executed,
                retrieval_entities,
                raw_retrieval_top_k,
                round_id,
                claim_topics,
                self.lexical_index,
                self.log_manager,
                post_text,
                claim_anchors=must_have_aliases or retrieval_entities,
                include_metrics=True,
            )
            debug_counts["kg_raw"] = int(retrieval_metrics.get("kg_raw", len(kg_candidates)))
            debug_counts["kg_with_score"] = int(retrieval_metrics.get("kg_with_score", 0))
            debug_counts["sem_raw"] = int(retrieval_metrics.get("sem_raw", 0))
            debug_counts["sem_filtered"] = int(retrieval_metrics.get("sem_filtered", len(dedup_sem)))
            await debug_reporter.log_step(
                step_name="Outputs of each VB/KG retrieval",
                description="Retrieval rerun after ingestion for current query",
                input_data={"query_index": query_idx + 1, "queries_executed": queries_executed},
                output_data={
                    "semantic_candidates_count": len(dedup_sem),
                    "kg_candidates_count": len(kg_candidates),
                    "semantic_candidates": dedup_sem,
                    "kg_candidates": kg_candidates,
                },
            )

            top_ranked = await rank_candidates(
                dedup_sem,
                kg_candidates,
                retrieval_entities,
                post_text,
                internal_top_k,
                round_id,
                self.log_manager,
                must_have_entities=must_have_aliases or retrieval_entities[:1],
            )
            debug_counts["kg_in_ranked"] = sum(1 for r in top_ranked if float(r.get("kg_score", 0.0) or 0.0) > 0.0)
            debug_counts["sem_in_ranked"] = sum(1 for r in top_ranked if float(r.get("sem_score", 0.0) or 0.0) > 0.0)
            await _emit_stage(
                "ranking_done",
                {
                    "query_index": query_idx + 1,
                    "ranked_count": len(top_ranked),
                    "kg_in_ranked": debug_counts["kg_in_ranked"],
                    "sem_in_ranked": debug_counts["sem_in_ranked"],
                },
            )
            self._log_top_domains(round_id, top_ranked)

            # Compute trust scores
            top_ranked_evidence = self._build_evidence_items(post_text, top_ranked)

            current_snapshot_id = self._evidence_snapshot_id_from_ranked(post_text, top_ranked)
            if current_snapshot_id in adaptive_cache:
                adaptive_trust = adaptive_cache[current_snapshot_id]
            else:
                adaptive_trust = self.trust_ranker.compute_adaptive_post_trust(
                    post_text, top_ranked_evidence, internal_top_k
                )
                adaptive_cache[current_snapshot_id] = adaptive_trust
            is_sufficient = adaptive_trust["is_sufficient"]

            logger.info(
                f"[CorrectivePipeline:{round_id}] After query {query_idx + 1}: "
                f"adaptive_sufficient={is_sufficient}, "
                f"trust_post={adaptive_trust['trust_post']:.3f}, "
                f"coverage={adaptive_trust['coverage']:.2f}, "
                f"total_facts={len(all_facts)}, "
                f"search_calls={search_api_calls}"
            )

            if is_comparative_claim:
                diversity_now = float(adaptive_trust.get("diversity", 0.0) or 0.0)
                comparative_hits = self._count_comparative_evidence_hits(post_text, top_ranked)
                if diversity_now <= 0.10 and comparative_hits <= best_comparative_hits:
                    comparative_monopoly_rounds += 1
                else:
                    comparative_monopoly_rounds = 0
                    best_comparative_hits = max(best_comparative_hits, comparative_hits)
                logger.info(
                    "[CorrectivePipeline:%s] Comparative quality: hits=%d best_hits=%d "
                    "diversity=%.2f monopoly_rounds=%d",
                    round_id,
                    comparative_hits,
                    best_comparative_hits,
                    diversity_now,
                    comparative_monopoly_rounds,
                )
                if comparative_monopoly_rounds >= 2:
                    logger.info(
                        "[CorrectivePipeline:%s] Comparative early-stop: domain monopoly persisted "
                        "(diversity<=0.10) with no new comparator evidence for %d rounds.",
                        round_id,
                        comparative_monopoly_rounds,
                    )
                    stop_reason = "comparative_monopoly"
                    break
                strong_covered_now = int(adaptive_trust.get("strong_covered", 0) or 0)
                trust_post_now = float(adaptive_trust.get("trust_post", 0.0) or 0.0)
                if (
                    search_api_calls >= 4
                    and strong_covered_now == 0
                    and comparative_hits <= 2
                    and trust_post_now < 0.40
                ):
                    logger.info(
                        "[CorrectivePipeline:%s] Comparative early-stop: saturated low-signal search "
                        "(calls=%d hits=%d strong_covered=%d trust_post=%.3f).",
                        round_id,
                        search_api_calls,
                        comparative_hits,
                        strong_covered_now,
                        trust_post_now,
                    )
                    stop_reason = "comparative_low_signal_saturation"
                    break

            # Step 6: Check if adaptive trust sufficient - STOP to save quota!
            if is_sufficient:
                remaining_queries = len(queries) - query_idx - 1
                logger.info(
                    f"[CorrectivePipeline:{round_id}] ADAPTIVE THRESHOLD MET after query {query_idx + 1}! "
                    f"Saved {remaining_queries} search API calls!"
                )
                stop_reason = "adaptive_sufficient"
                break

            if confidence_mode:
                gain_estimate = self._estimate_information_gain(
                    adaptive_trust,
                    zero_extraction_rounds=zero_extraction_rounds,
                    low_yield_rounds=low_yield_rounds,
                )
                should_stop, stop_reason = self._should_stop_confidence_mode(
                    claim_frame=claim_frame,
                    adaptive_trust=adaptive_trust,
                    confidence_target_coverage=confidence_target_coverage,
                    confidence_max_new_trusted_urls=confidence_max_new_trusted_urls,
                    new_trusted_urls_processed=new_trusted_urls_processed,
                    zero_extraction_rounds=zero_extraction_rounds,
                    low_yield_rounds=low_yield_rounds,
                )
                logger.info("[CorrectivePipeline:%s] Confidence mode decision: %s", round_id, stop_reason)
                if should_stop:
                    logger.info("[CorrectivePipeline:%s] Confidence mode stop: %s", round_id, stop_reason)
                    break

        # Log final statistics
        logger.info(
            f"[CorrectivePipeline:{round_id}] Completed: {len(all_facts)} facts, "
            f"{len(top_ranked)} ranked, {search_api_calls}/{len(queries)} queries used, "
            f"{len(processed_urls)} URLs processed, {len(skipped_urls)} URLs skipped (already processed)"
        )

        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Pipeline completed: {len(all_facts)} facts, {search_api_calls} API calls",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={
                    "facts_count": len(all_facts),
                    "triples_count": len(all_triples),
                    "ranked_count": len(top_ranked),
                    "top_score": top_ranked[0]["final_score"] if top_ranked else 0.0,
                    "urls_processed": len(processed_urls),
                    "urls_skipped": len(skipped_urls),
                    "search_api_calls": search_api_calls,
                    "queries_total": len(queries),
                    "queries_saved": len(queries) - search_api_calls,
                },
            )

        # Determine final status
        final_status = "completed"
        if not all_facts and search_api_calls > 0 and not top_ranked:
            final_status = "no_facts_extracted"
        elif search_api_calls == 0 and not top_ranked:
            final_status = "no_search_results"

        # ====================================================================
        # PHASE 7: RAG Generation - Generate Final Verdict
        # ====================================================================
        logger.info(f"[CorrectivePipeline:{round_id}] Generating verdict with RAG...")

        # Compute final trust scores
        final_top_ranked_evidence = self._build_evidence_items(post_text, top_ranked)

        final_snapshot_id = self._evidence_snapshot_id_from_ranked(post_text, top_ranked)
        final_adaptive_trust = adaptive_cache.get(final_snapshot_id)
        if final_adaptive_trust is None:
            final_adaptive_trust = self.trust_ranker.compute_adaptive_post_trust(
                post_text, final_top_ranked_evidence, internal_top_k
            )
            adaptive_cache[final_snapshot_id] = final_adaptive_trust
        else:
            logger.info(
                "[CorrectivePipeline:%s] Using memoized adaptive trust snapshot=%s trust_post=%.3f sufficient=%s",
                round_id,
                final_snapshot_id[:12],
                float(final_adaptive_trust.get("trust_post", 0.0) or 0.0),
                bool(final_adaptive_trust.get("is_sufficient", False)),
            )
        if gain_estimate <= 0.0:
            gain_estimate = self._estimate_information_gain(
                final_adaptive_trust,
                zero_extraction_rounds=zero_extraction_rounds,
                low_yield_rounds=low_yield_rounds,
            )
        final_trust_post = self.trust_ranker.compute_post_trust(final_top_ranked_evidence, internal_top_k)
        trust_snapshot_v2 = self.trust_ranker.compute_uncertainty_snapshot(final_top_ranked_evidence, internal_top_k)
        trust_snapshot_v2["sufficiency_reason"] = str(final_adaptive_trust.get("gate_reason") or "")
        trust_snapshot_v2["info_gain"] = round(float(gain_estimate or 0.0), 4)
        final_trust_score = final_trust_post["trust_post"]
        adaptive_payload = self._trust_payload_adaptive(final_adaptive_trust)
        fixed_payload = self._trust_payload_fixed(final_trust_score, self.CONF_THRESHOLD)

        verdict_result = await self.verdict_generator.generate_verdict(
            claim=post_text,
            ranked_evidence=top_ranked,
            top_k=internal_top_k,
            used_web_search=search_api_calls > 0,
            adaptive_metrics=final_adaptive_trust,
            evidence_snapshot_id=final_snapshot_id,
        )
        # Final trust harmonization:
        # force the verdict payload to respect the final adaptive trust gate used by pipeline output.
        verdict_with_trust = dict(verdict_result or {})
        verdict_with_trust["trust_threshold_met"] = bool(adaptive_payload.get("trust_threshold_met", False))
        verdict_result = self.verdict_generator._enforce_binary_verdict_payload(
            post_text,
            verdict_with_trust,
            evidence=top_ranked,
        )

        logger.info(
            f"[CorrectivePipeline:{round_id}] Final verdict: {verdict_result['verdict']} "
            f"(confidence: {verdict_result['confidence']:.2f})"
        )

        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Verdict generated: {verdict_result['verdict']}",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={
                    "verdict": verdict_result["verdict"],
                    "confidence": verdict_result["confidence"],
                    "rationale": verdict_result.get("rationale", ""),
                },
            )
        await _emit_stage(
            "completed",
            {
                "status": final_status,
                "search_api_calls": search_api_calls,
                "stop_reason": stop_reason,
                "gain_estimate": round(float(gain_estimate or 0.0), 4),
                "kg_timeout_count": kg_timeout_count,
                "zero_extraction_rounds": zero_extraction_rounds,
            },
        )
        await debug_reporter.log_step(
            step_name="Pipeline completed",
            description="Final ranked output and verdict summary",
            input_data={"queries_executed": queries_executed, "search_api_calls": search_api_calls},
            output_data={
                "status": final_status,
                "ranked_count": len(top_ranked),
                "verdict": verdict_result.get("verdict"),
                "confidence": verdict_result.get("confidence"),
            },
        )
        await debug_reporter.close()

        output_facts = list(all_facts or [])
        if not output_facts and top_ranked:
            # Keep downstream consumers from receiving an empty fact list when
            # we still have admissible ranked evidence.
            output_facts = [
                {
                    "statement": str(item.get("statement") or ""),
                    "source_url": str(item.get("source_url") or ""),
                    "source": str(item.get("source") or ""),
                    "confidence": float(item.get("final_score", 0.0) or 0.0),
                }
                for item in top_ranked
                if str(item.get("statement") or "").strip()
            ]
        output_triples = list(all_triples or [])
        if not output_triples and top_ranked:
            for item in top_ranked:
                entities = [str(e).strip() for e in (item.get("entities") or []) if str(e).strip()]
                subject = str(item.get("subject") or (entities[0] if entities else "")).strip()
                object_ = str(item.get("object") or (entities[1] if len(entities) > 1 else "")).strip()
                relation = str(item.get("relation") or "related_to").strip()
                if not subject or not object_:
                    continue
                output_triples.append(
                    {
                        "id": str(uuid.uuid4()),
                        "subject": subject,
                        "relation": relation,
                        "object": object_,
                        "confidence": float(item.get("final_score", 0.0) or 0.0),
                        "source_url": str(item.get("source_url") or ""),
                    }
                )
                if len(output_triples) >= 3:
                    break

        return {
            "round_id": round_id,
            "status": final_status,
            "facts": output_facts,
            "triples": output_triples,
            "queries": queries_executed,
            "queries_total": len(queries),
            "semantic_candidates_count": len(dedup_sem),
            "kg_candidates_count": len(kg_candidates),
            "ranked": top_ranked,
            "used_web_search": search_api_calls > 0,
            "data_source": "WEB_SEARCH" if search_api_calls > 0 else "CACHE",
            "search_api_calls": search_api_calls,
            "search_api_calls_saved": len(queries) - search_api_calls,
            "urls_processed": len(processed_urls),
            "urls_skipped_already_processed": len(skipped_urls),
            "initial_top_score": final_trust_score,
            "ranking_top_score": (top_ranked[0].get("final_score", 0.0) if top_ranked else 0.0),
            "ranking_avg_score": (
                sum(float(r.get("final_score", 0.0) or 0.0) for r in top_ranked[:5]) / max(1, len(top_ranked[:5]))
                if top_ranked
                else 0.0
            ),
            "trust_post": final_adaptive_trust["trust_post"],
            "trust_post_adaptive": final_adaptive_trust["trust_post"],
            "trust_post_fixed": final_trust_score,
            "trust_snapshot_v2": trust_snapshot_v2,
            "trust_grade": final_trust_post.get("grade", "D"),
            "agreement_ratio": final_adaptive_trust.get("agreement", final_trust_post.get("agreement_ratio", 0.0)),
            "coverage": final_adaptive_trust.get("coverage", 0.0),
            "diversity": final_adaptive_trust.get("diversity", 0.0),
            "num_subclaims": final_adaptive_trust.get("num_subclaims", 1),
            "adaptive_is_sufficient": bool(final_adaptive_trust.get("is_sufficient", False)),
            "fixed_trust_threshold": self.CONF_THRESHOLD,
            "fixed_trust_threshold_met": bool(fixed_payload.get("trust_threshold_met", False)),
            "verdict": verdict_result,
            "debug_counts": debug_counts,
            "vdb_signal_count": debug_counts["sem_in_ranked"],
            "kg_signal_count": debug_counts["kg_in_ranked"],
            "vdb_signal_sum_top5": round(
                sum(float(e.get("sem_score", 0.0) or 0.0) for e in top_ranked[:5]),
                3,
            ),
            "kg_signal_sum_top5": round(
                sum(float(e.get("kg_score", 0.0) or 0.0) for e in top_ranked[:5]),
                3,
            ),
            "llm": get_groq_job_metadata(),
            "domain": normalized_domain,
            "health_scope": {
                "in_scope": health_scope.health_in_scope,
                "biomedical_confidence": health_scope.biomedical_confidence,
                "scope_reason": health_scope.scope_reason,
            },
            "pipeline_diagnostics_v2": {
                "stop_reason": stop_reason,
                "gain_estimate": round(float(gain_estimate or 0.0), 4),
                "kg_timeout_count": int(kg_timeout_count),
                "zero_extraction_rounds": int(zero_extraction_rounds),
            },
            **adaptive_payload,
        }

    @staticmethod
    def _entity_tokenize(text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", text or "")]

    @staticmethod
    def _is_high_salience_entity(entity: str) -> bool:
        low = str(entity or "").strip().lower()
        if not low:
            return False
        if re.search(r"\b(that|which|who|whom|whose)\b", low):
            return False
        generic = {
            "anatomical terms",
            "medical terms",
            "health terms",
            "general",
            "unknown",
            "misc",
            "medicine",
            "medicines",
            "herbal",
            "sleep",
            "used",
            "aid",
            "effect",
            "effects",
            "study",
            "research",
            "evidence",
            "thing",
            "things",
            "factor",
            "factors",
            "element",
            "elements",
            "chemical",
            "chemicals",
            "compound",
            "compounds",
            "mineral",
            "minerals",
            "nutrient",
            "nutrients",
            "substance",
            "substances",
            "agent",
            "agents",
            "product",
            "products",
            "system",
            "process",
            "condition",
        }
        verbs = {
            "is",
            "are",
            "was",
            "were",
            "be",
            "being",
            "been",
            "use",
            "used",
            "using",
            "help",
            "helps",
            "support",
            "supports",
            "cause",
            "causes",
            "reduce",
            "improve",
            "prevent",
            "prevents",
            "prevented",
            "preventing",
            "treat",
            "treats",
            "treated",
            "treating",
            "cure",
            "cures",
            "cured",
            "curing",
            "boost",
            "boosts",
            "boosted",
            "boosting",
            "stop",
            "stops",
            "stopped",
            "stopping",
        }
        if low in generic or low in verbs:
            return False
        toks = CorrectivePipeline._entity_tokenize(low)
        if not toks:
            return False
        informative = [t for t in toks if t not in generic and t not in verbs]
        if not informative:
            return False
        if len(toks) == 1 and toks[0] in generic:
            return False
        return True

    @staticmethod
    def _normalize_entity_phrase(entity: str) -> str:
        low = str(entity or "").strip().lower()
        if not low:
            return ""
        low = re.sub(r"^(and|or|but)\s+", "", low)
        low = re.sub(r"\b(?:that|which|who|whom|whose)\b.*$", "", low).strip(" ,.")
        tokens = CorrectivePipeline._entity_tokenize(low)
        if not tokens:
            return ""
        noisy = {
            "necessary",
            "normal",
            "overall",
            "general",
            "effect",
            "effects",
            "evidence",
            "study",
            "research",
            "used",
            "use",
            "using",
            "prevent",
            "prevents",
            "prevented",
            "preventing",
            "treat",
            "treats",
            "treated",
            "treating",
            "cure",
            "cures",
            "cured",
            "curing",
            "boost",
            "boosts",
            "boosted",
            "boosting",
            "stop",
            "stops",
            "stopped",
            "stopping",
        }
        cleaned = [t for t in tokens if t not in noisy]
        if not cleaned:
            return ""
        return " ".join(cleaned[:4]).strip()

    def _select_must_have_entities(self, claim: str, entities: List[str]) -> List[str]:
        if not claim:
            return entities[:1]
        claim_tokens = self._entity_tokenize(claim)
        lead_window = max(1, int(len(claim_tokens) * 0.5))
        lead_tokens = set(claim_tokens[:lead_window])
        candidates = [self._normalize_entity_phrase(e) for e in entities]
        candidates = [e for e in candidates if self._is_high_salience_entity(e)]
        if not candidates:
            candidates = [self._normalize_entity_phrase(e) for e in entities]
            candidates = [e for e in candidates if e]
        scored: List[tuple[int, int, str]] = []
        claim_low = claim.lower()
        broad_nouns = {
            "mineral",
            "minerals",
            "chemical",
            "chemicals",
            "nutrient",
            "nutrients",
            "compound",
            "compounds",
            "substance",
            "substances",
        }
        weak_singletons = {
            "adult",
            "adults",
            "human",
            "body",
            "located",
            "more",
            "than",
            "half",
            "most",
            "many",
            "some",
        }
        candidate_token_sets = [(c, set(self._entity_tokenize(c))) for c in candidates if c]
        for e in candidates:
            text = str(e).strip()
            toks = self._entity_tokenize(text)
            if not toks:
                continue
            score = 0
            if text and re.search(rf"\b{re.escape(text.lower())}\b", claim_low):
                score += 4
            if any(t in lead_tokens for t in toks):
                score += 2
            if len(toks) == 1:
                score += 2
            elif len(toks) == 2:
                score += 1
            else:
                score -= 1
            # Prefer specific entities over generic heads (e.g., "vitamin c" over "vitamin").
            toks_set = set(toks)
            if len(toks_set) == 1:
                tok = next(iter(toks_set), "")
                if tok in weak_singletons:
                    score -= 4
                has_more_specific_variant = any(
                    other_text != text and toks_set and toks_set < other_toks
                    for other_text, other_toks in candidate_token_sets
                )
                if has_more_specific_variant:
                    score -= 3
            if set(toks) <= broad_nouns:
                score -= 3
            scored.append((score, len(text), text))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return [scored[0][2]] if scored else []

    def _build_must_have_aliases(self, claim_entities: List[str], must_have_entities: List[str]) -> List[str]:
        aliases: List[str] = []
        for e in must_have_entities or []:
            n = self._normalize_entity_phrase(e)
            if n and self._is_high_salience_entity(n) and n not in aliases:
                aliases.append(n)

        core_tokens = set(self._entity_tokenize(aliases[0])) if aliases else set()
        for e in claim_entities or []:
            n = self._normalize_entity_phrase(e)
            if not n or not self._is_high_salience_entity(n) or n in aliases:
                continue
            toks = set(self._entity_tokenize(n))
            if core_tokens and not (toks & core_tokens):
                continue
            aliases.append(n)
            if len(aliases) >= 3:
                break

        if not aliases:
            for e in claim_entities or []:
                n = self._normalize_entity_phrase(e)
                if n and self._is_high_salience_entity(n):
                    aliases.append(n)
                    break
        return aliases

    async def _extract_claim_entities(self, claim: str, round_id: str) -> List[str]:
        """Extract high-salience claim entities with deterministic noun-phrase fallback."""
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
            "over",
            "under",
            "about",
            "around",
            "average",
            "per",
            "times",
            "not",
            "no",
            "never",
            "do",
            "does",
            "did",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "itself",
            "through",
            "scientifically",
            "supported",
            "support",
            "supports",
        }

        def _fallback_entities(text: str) -> List[str]:
            blocked_tokens = {
                "not",
                "no",
                "never",
                "scientifically",
                "supported",
                "support",
                "supports",
                "proven",
                "prove",
                "proves",
            }
            leading_verb_tokens = {
                "cause",
                "causes",
                "prevent",
                "prevents",
                "treat",
                "treats",
                "cure",
                "cures",
                "boost",
                "boosts",
                "help",
                "helps",
                "support",
                "supports",
            }

            def _valid_candidate(raw: str) -> str:
                candidate = self._normalize_entity_phrase(raw)
                if not candidate or not self._is_high_salience_entity(candidate):
                    return ""
                toks = [t for t in self._entity_tokenize(candidate) if t]
                if not toks:
                    return ""
                if any(t in blocked_tokens for t in toks):
                    return ""
                if len(toks) > 1 and toks[0] in leading_verb_tokens:
                    return ""
                return candidate

            phrases = re.findall(
                r"\b([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){0,2})\b",
                text or "",
            )
            preferred: List[str] = []

            # Prefer clean atomic tokens first to avoid synthetic bigrams crowding out real entities.
            unigram_tokens = [t for t in self._entity_tokenize(text) if t not in stop]
            for t in unigram_tokens:
                normalized = _valid_candidate(t)
                if normalized and normalized not in preferred:
                    preferred.append(normalized)

            for phrase in phrases:
                low = phrase.strip().lower()
                toks = [t for t in self._entity_tokenize(low) if t not in stop]
                if len(toks) < 2:
                    continue
                candidate = _valid_candidate(" ".join(toks))
                if candidate and candidate not in preferred:
                    preferred.append(candidate)
            return preferred[:10]

        llm_entities: List[str] = []
        try:
            synthetic_facts = [{"statement": claim, "source_url": "claim", "fact_id": "claim_0"}]
            annotated = await self.entity_extractor.annotate_entities(synthetic_facts)
            if annotated:
                entities = annotated[0].get("entities", []) or []
                cleaned = [str(e).strip() for e in entities if str(e).strip()]
                llm_entities = [e for e in cleaned if self._is_high_salience_entity(e)]
        except Exception as e:
            logger.warning(f"[CorrectivePipeline:{round_id}] Entity extraction from claim failed: {e}")

        fallback = _fallback_entities(claim)
        merged: List[str] = []
        for e in llm_entities + fallback:
            low = str(e).strip().lower()
            if low and low not in merged and self._is_high_salience_entity(low):
                merged.append(low)

        if fallback and fallback != llm_entities:
            logger.info(f"[CorrectivePipeline:{round_id}] Fallback entities from claim: {fallback}")
            if self.log_manager:
                await self.log_manager.add_log(
                    level="INFO",
                    message=f"Fallback entities from claim: {fallback}",
                    module=__name__,
                    request_id=f"claim-{round_id}",
                    round_id=round_id,
                    context={"fallback_entities": fallback},
                )
        return merged or fallback

    def _build_retrieval_queries(self, claim: str) -> List[str]:
        """
        Build retrieval queries using the full claim plus a few decomposed subclaims.
        This increases recall for multi-part claims while keeping VDB calls bounded.
        """
        queries = [claim.strip()] if claim and claim.strip() else []

        def _tokens(text: str) -> List[str]:
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
            return [w for w in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", (text or "").lower()) if w not in stop]

        def _predicate_object_hint(text: str) -> str:
            low = str(text or "").strip()
            if not low:
                return ""
            patterns = (
                r"^(?P<subj>.+?)\s+\b(?:is|are|was|were)\b\s+(?P<pred_obj>.+)$",
                (
                    r"^(?P<subj>.+?)\s+\b(?:contribut(?:e|es|ed|ing)|help|helps|support|supports|"
                    r"reduce|reduces|prevent|prevents|cause|causes)\b\s+(?P<pred_obj>.+)$"
                ),
            )
            for pattern in patterns:
                m = re.search(pattern, low, flags=re.IGNORECASE)
                if not m:
                    continue
                subj = " ".join(_tokens(m.group("subj")))
                pred_obj = " ".join(_tokens(m.group("pred_obj")))
                if subj and pred_obj:
                    return f"{subj} {pred_obj}".strip()
            return ""

        subclaims = self.trust_ranker.adaptive_policy.decompose_claim(claim)
        # Limit to top 3 subclaims to avoid query explosion
        for sub in subclaims[:3]:
            s = sub.strip()
            if s and s not in queries:
                queries.append(s)
            hint = _predicate_object_hint(s)
            if hint and hint not in queries:
                queries.append(hint)
        return queries

    async def _infer_claim_topics(self, claim: str, entities: List[str]) -> tuple[list[str], float]:
        subclaims = self.trust_ranker.adaptive_policy.decompose_claim(claim)
        topics: set[str] = set()
        best_conf = 0.0
        for sub in subclaims:
            t, conf = await self.topic_classifier.classify(sub, entities, None)
            for item in t:
                topics.add(item)
            if conf > best_conf:
                best_conf = conf
        return list(topics), best_conf


__all__ = ["CorrectivePipeline"]
