from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

import app.services.corrective.pipeline as pipeline_module
from app.services.corrective.pipeline import CorrectivePipeline


class _DummyCanonicalClaim:
    canonical_accept_rate = 1.0

    def __init__(self, claim: str) -> None:
        self.claim = claim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_original": self.claim,
            "segments": [],
            "canonical_accept_rate": 1.0,
            "canonical_parse_failed": False,
            "canonical_failure_reason": "",
        }


class _DummyCanonicalizer:
    async def canonicalize_claim(self, claim: str) -> _DummyCanonicalClaim:
        return _DummyCanonicalClaim(claim)

    def split_query_tracks(self, _canonical_claim: _DummyCanonicalClaim) -> tuple[List[str], List[str]]:
        return [], []

    def build_dual_track_queries(
        self, _canonical_claim: _DummyCanonicalClaim, max_per_segment: int = 8
    ) -> Dict[str, Any]:
        _ = max_per_segment
        return {"queries_original": [], "queries_canonical": [], "query_facets": []}


class _DummyAdaptivePolicy:
    def decompose_claim(self, claim: str) -> List[str]:
        return [claim]


class _DummyTrustRanker:
    def __init__(self) -> None:
        self.adaptive_policy = _DummyAdaptivePolicy()

    def classify_stance_for_evidence(self, claim: str, evidence: List[Any]) -> None:
        _ = (claim, evidence)

    def compute_adaptive_post_trust(self, claim: str, evidence: List[Any], top_k: int) -> Dict[str, Any]:
        _ = (claim, evidence, top_k)
        return {
            "is_sufficient": False,
            "coverage": 0.0,
            "diversity": 0.0,
            "agreement": 0.0,
            "trust_post": 0.0,
            "num_subclaims": 1,
            "strong_covered": 0,
            "gate_reason": "insufficient",
            "contradicted_subclaims": 0,
        }

    def compute_post_trust(self, evidence: List[Any], top_k: int) -> Dict[str, Any]:
        _ = (evidence, top_k)
        return {"trust_post": 0.0, "grade": "D", "agreement_ratio": 0.0}

    def compute_uncertainty_snapshot(self, evidence: List[Any], top_k: int) -> Dict[str, Any]:
        _ = (evidence, top_k)
        return {"uncertainty": 1.0}


class _DummySearchAgent:
    def __init__(self) -> None:
        self.generate_search_queries = AsyncMock(return_value=[])
        self.execute_single_query = AsyncMock(return_value=[])
        self.search_for_claim = AsyncMock(return_value=[])


class _DummyScraper:
    def reset_job_attempts(self) -> None:
        return None


class _DummyVDBIngest:
    def get_processed_urls(self, topics: List[str], max_urls: int) -> set[str]:
        _ = (topics, max_urls)
        return set()


class _DummyVerdictGenerator:
    async def generate_verdict(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return {
            "verdict": "UNVERIFIABLE",
            "verdict_internal": "UNVERIFIABLE",
            "confidence": 0.05,
            "truthfulness_percent": 50.0,
            "truth_score_binary": 0.5,
            "rationale": "insufficient directional evidence",
            "support_mass": 0.0,
            "contradict_mass": 0.0,
            "neutral_mass": 0.0,
            "policy_trace": [],
        }

    def _enforce_binary_verdict_payload(
        self, claim: str, payload: Dict[str, Any], evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        _ = (claim, evidence)
        return payload


class _DummyDebugReporter:
    def __init__(self, run_id: str) -> None:
        _ = run_id

    async def initialize(self, claim: str) -> None:
        _ = claim

    async def log_step(self, **kwargs: Any) -> None:
        _ = kwargs

    async def close(self) -> None:
        return None


def _build_pipeline() -> CorrectivePipeline:
    pipeline = CorrectivePipeline.__new__(CorrectivePipeline)
    pipeline.search_agent = _DummySearchAgent()
    pipeline.scraper = _DummyScraper()
    pipeline.fact_extractor = object()
    pipeline.entity_extractor = object()
    pipeline.relation_extractor = object()
    pipeline.vdb_ingest = _DummyVDBIngest()
    pipeline.vdb_retriever = object()
    pipeline.kg_retriever = object()
    pipeline.lexical_index = None
    pipeline.topic_classifier = object()
    pipeline.verdict_generator = _DummyVerdictGenerator()
    pipeline.claim_canonicalizer = _DummyCanonicalizer()
    pipeline.trust_ranker = _DummyTrustRanker()
    pipeline.log_manager = None
    pipeline._extract_claim_entities = AsyncMock(return_value=["magnesium", "bracelets", "migraines"])
    pipeline._infer_claim_topics = AsyncMock(return_value=(["pathology"], 0.8))
    return pipeline


@pytest.mark.asyncio
async def test_pipeline_no_queries_emits_timing_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline()
    pipeline.search_agent.generate_search_queries = AsyncMock(return_value=[])
    monkeypatch.setenv("LUXIA_CONFIDENCE_MODE", "true")

    async def _fake_retrieve_candidates(
        *args: Any, **kwargs: Any
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        _ = (args, kwargs)
        return (
            [],
            [],
            {
                "sem_raw": 0,
                "sem_filtered": 0,
                "kg_raw": 0,
                "kg_with_score": 0,
                "semantic_latency_seconds": 0.01,
                "kg_latency_seconds": 0.02,
                "kg_zero_signal": 1,
                "kg_fallback_triggered": 0,
            },
        )

    monkeypatch.setattr(pipeline_module, "retrieve_candidates", _fake_retrieve_candidates)
    monkeypatch.setattr(pipeline_module, "rank_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(pipeline_module, "PipelineDebugReporter", _DummyDebugReporter)
    monkeypatch.setattr(pipeline_module, "reset_groq_counter", lambda job_id=None: None)

    result = await pipeline.run("Magnesium bracelets cure chronic migraines.", "health", top_k=3)
    profile = result["timing_profile"]

    assert result["status"] == "no_queries_generated"
    assert result["verdict"]["verdict"] == "UNVERIFIABLE"
    required_fields = {
        "total_latency_seconds",
        "intake_latency_seconds",
        "claim_understanding_latency_seconds",
        "vector_retrieval_latency_seconds",
        "kg_retrieval_latency_seconds",
        "initial_ranking_latency_seconds",
        "corrective_round_count",
        "corrective_total_latency_seconds",
        "corrective_round_latencies",
        "web_search_latency_seconds_total",
        "scrape_latency_seconds_total",
        "extraction_latency_seconds_total",
        "rerank_latency_seconds_total",
        "final_verdict_latency_seconds",
        "stage_callback_latency_seconds_total",
        "unique_queries_generated",
        "unique_queries_executed",
        "duplicate_queries_skipped",
        "urls_considered",
        "urls_fetched",
        "duplicate_urls_skipped",
        "extracted_docs_count",
        "duplicate_docs_skipped",
        "zero_yield_round_count",
        "new_directional_evidence_by_round",
        "stop_reason",
    }
    assert required_fields.issubset(set(profile.keys()))


@pytest.mark.asyncio
async def test_pipeline_duplicate_counters_do_not_change_baseline_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline()
    pipeline.search_agent.generate_search_queries = AsyncMock(side_effect=[["q1", "q1", "q2"], []])
    pipeline.search_agent.execute_single_query = AsyncMock(
        side_effect=[
            [
                "https://example.org/a?utm_source=test",
                "https://example.org/a",
                "https://example.org/b",
            ],
            [
                "https://example.org/c?utm_campaign=z",
                "https://example.org/c",
                "https://example.org/d",
            ],
        ]
    )
    pipeline.search_agent.search_for_claim = AsyncMock(
        return_value=[
            "https://example.org/a?utm_source=test",
            "https://example.org/a",
            "https://example.org/b",
        ],
    )
    monkeypatch.setenv("LUXIA_CONFIDENCE_MODE", "true")

    async def _fake_retrieve_candidates(
        *args: Any, **kwargs: Any
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        _ = (args, kwargs)
        return (
            [],
            [],
            {
                "sem_raw": 0,
                "sem_filtered": 0,
                "kg_raw": 0,
                "kg_with_score": 0,
                "semantic_latency_seconds": 0.01,
                "kg_latency_seconds": 0.01,
                "kg_zero_signal": 1,
                "kg_fallback_triggered": 0,
            },
        )

    async def _fake_scrape_pages(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        urls = args[1] if len(args) > 1 else []
        _ = kwargs
        if any("/c" in str(url) or "/d" in str(url) for url in urls):
            content = "second doc body"
            base = "https://example.org/c"
        else:
            content = "same doc body"
            base = "https://example.org/a"
        return [
            {"url": base, "content": content, "source": "example.org"},
            {"url": f"{base}?utm_source=test", "content": content, "source": "example.org"},
        ]

    async def _fake_extract_all(
        *args: Any, **kwargs: Any
    ) -> tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        _ = (args, kwargs)
        return [], [], []

    monkeypatch.setattr(pipeline_module, "retrieve_candidates", _fake_retrieve_candidates)
    monkeypatch.setattr(pipeline_module, "rank_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(pipeline_module, "scrape_pages", _fake_scrape_pages)
    monkeypatch.setattr(pipeline_module, "extract_all", _fake_extract_all)
    monkeypatch.setattr(
        pipeline_module,
        "ingest_facts_and_triples",
        AsyncMock(return_value={"kg_timeout_count": 0}),
    )
    monkeypatch.setattr(pipeline_module, "PipelineDebugReporter", _DummyDebugReporter)
    monkeypatch.setattr(pipeline_module, "reset_groq_counter", lambda job_id=None: None)

    result = await pipeline.run("Magnesium bracelets cure chronic migraines.", "health", top_k=3)
    profile = result["timing_profile"]

    assert result["pipeline_diagnostics_v2"]["stop_reason"] == "query_budget_exhausted"
    assert profile["duplicate_queries_skipped"] >= 1
    assert profile["duplicate_urls_skipped"] >= 1
    assert profile["duplicate_docs_skipped"] >= 1
    assert profile["unique_queries_executed"] == 2
    assert result["queries_used"] == 2
    assert profile["stop_reason"] == "query_budget_exhausted"


@pytest.mark.asyncio
async def test_stage_callback_latency_is_measured_with_blocking_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _build_pipeline()
    pipeline.search_agent.generate_search_queries = AsyncMock(return_value=[])
    monkeypatch.setenv("LUXIA_CONFIDENCE_MODE", "true")

    async def _fake_retrieve_candidates(
        *args: Any, **kwargs: Any
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        _ = (args, kwargs)
        return (
            [],
            [],
            {
                "sem_raw": 0,
                "sem_filtered": 0,
                "kg_raw": 0,
                "kg_with_score": 0,
                "semantic_latency_seconds": 0.01,
                "kg_latency_seconds": 0.02,
                "kg_zero_signal": 1,
                "kg_fallback_triggered": 0,
            },
        )

    callback_stages: List[str] = []

    async def _stage_callback(stage: str, payload: Dict[str, Any]) -> None:
        _ = payload
        callback_stages.append(stage)
        await asyncio.sleep(0.01)

    monkeypatch.setattr(pipeline_module, "retrieve_candidates", _fake_retrieve_candidates)
    monkeypatch.setattr(pipeline_module, "rank_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(pipeline_module, "PipelineDebugReporter", _DummyDebugReporter)
    monkeypatch.setattr(pipeline_module, "reset_groq_counter", lambda job_id=None: None)

    result = await pipeline.run(
        "Magnesium bracelets cure chronic migraines.",
        "health",
        top_k=3,
        stage_callback=_stage_callback,
    )
    profile = result["timing_profile"]

    assert callback_stages[0] == "started"
    assert callback_stages[-1] == "completed"
    assert profile["stage_callback_latency_seconds_total"] > 0.0


def test_pipeline_url_and_directional_helpers() -> None:
    normalized = CorrectivePipeline._normalize_url_key("https://Example.org/path/?utm_source=x&id=1")
    assert normalized == "https://example.org/path?id=1"

    ranked = [
        {
            "source_url": "https://a",
            "statement": "e1",
            "stance": "contradicts",
            "support_score": 0.1,
            "contradict_score": 0.7,
        },
        {
            "source_url": "https://b",
            "statement": "e2",
            "stance": "neutral",
            "support_score": 0.2,
            "contradict_score": 0.2,
        },
    ]
    keys = CorrectivePipeline._directional_evidence_keys(ranked, min_directional_score=0.6)
    assert len(keys) == 1
