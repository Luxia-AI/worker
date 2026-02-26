import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from prometheus_client import Counter
from pydantic import BaseModel, Field
from shared.metrics import install_metrics

SERVICE_NAME = "worker"
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
SERVICE_ENV = os.getenv("APP_ENV", "prod")

rag_jobs_completed_total = Counter(
    "rag_jobs_completed_total",
    "Total completed RAG jobs",
)
rag_jobs_failed_total = Counter(
    "rag_jobs_failed_total",
    "Total failed RAG jobs",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Luxia Worker", version=SERVICE_VERSION)
install_metrics(app, service_name=SERVICE_NAME, version=SERVICE_VERSION, env=SERVICE_ENV)

_pipeline_lock = asyncio.Lock()
_pipeline: Any | None = None


class VerifyRequest(BaseModel):
    job_id: str
    claim: str
    room_id: str | None = None
    client_id: str | None = None
    client_claim_id: str | None = None
    source: str | None = None
    domain: str = Field(default="general")
    top_k: int = Field(default=5, ge=1, le=20)


def _mock_verdict_for_claim(claim: str) -> tuple[str, float, float, str]:
    claim_l = claim.lower()
    if any(k in claim_l for k in ("hoax", "fake", "myth")):
        return ("MISLEADING", 0.86, 22.0, "Claim contains patterns commonly associated with misinformation framing.")
    if any(k in claim_l for k in ("significantly", "guaranteed", "always", "never", "detoxifies")):
        return (
            "UNVERIFIABLE",
            0.64,
            48.0,
            "Claim makes strong causal assertions without attached evidence in this flow.",
        )
    if any(k in claim_l for k in ("improves", "supports", "is linked to", "can reduce")):
        return ("PARTIALLY_SUPPORTED", 0.71, 63.0, "Claim appears plausible but requires stronger source validation.")
    return ("UNVERIFIABLE", 0.58, 50.0, "Insufficient evidence context for deterministic verification.")


async def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    async with _pipeline_lock:
        if _pipeline is None:
            from app.services.corrective.pipeline import CorrectivePipeline

            _pipeline = CorrectivePipeline()
    return _pipeline


@app.on_event("startup")
async def preload_pipeline_on_startup() -> None:
    preload_enabled = os.getenv("WORKER_PRELOAD_PIPELINE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not preload_enabled:
        logger.info("[Worker] Startup preload disabled (WORKER_PRELOAD_PIPELINE=false)")
        return

    started = datetime.now(timezone.utc)
    logger.info("[Worker] Startup preload enabled, initializing pipeline")
    try:
        await _get_pipeline()
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        logger.info("[Worker] Pipeline preload complete in %.2fs", elapsed)
    except Exception:
        # Keep service booting; verify endpoint retains runtime fallback behavior.
        logger.exception("[Worker] Pipeline preload failed; runtime initialization will retry")


def _format_completed_response(payload: VerifyRequest, result: dict[str, Any]) -> dict[str, Any]:
    ranked_evidence = result.get("ranked", []) or []
    verdict_result = result.get("verdict", {}) or {}
    llm_meta = result.get("llm", {}) or {}
    status = str(result.get("status", "completed") or "completed")

    return {
        "status": "completed",
        "job_id": payload.job_id,
        "room_id": payload.room_id,
        "client_id": payload.client_id,
        "client_claim_id": payload.client_claim_id,
        "claim": payload.claim,
        "pipeline_status": status,
        "result_status": status,
        "verdict": verdict_result.get("verdict", "UNVERIFIABLE"),
        "verdict_confidence": verdict_result.get("confidence", 0.0),
        "truthfulness_percent": verdict_result.get("truthfulness_percent", 0.0),
        "verdict_rationale": verdict_result.get("rationale", ""),
        "key_findings": verdict_result.get("key_findings", []),
        "claim_breakdown": verdict_result.get("claim_breakdown", []),
        "evidence_map": verdict_result.get("evidence_map", []),
        "evidence_count": len(ranked_evidence),
        "facts_extracted": len(result.get("facts", []) or []),
        "semantic_candidates_count": int(result.get("semantic_candidates_count", 0) or 0),
        "kg_candidates_count": int(result.get("kg_candidates_count", 0) or 0),
        "vdb_signal_count": int(result.get("vdb_signal_count", 0) or 0),
        "kg_signal_count": int(result.get("kg_signal_count", 0) or 0),
        "vdb_signal_sum_top5": float(result.get("vdb_signal_sum_top5", 0) or 0),
        "kg_signal_sum_top5": float(result.get("kg_signal_sum_top5", 0) or 0),
        "top_ranking_score": float(result.get("ranking_top_score", 0) or 0),
        "avg_ranking_score": float(result.get("ranking_avg_score", 0) or 0),
        "trust_policy_mode": result.get("trust_policy_mode", "fixed"),
        "trust_metric_name": result.get("trust_metric_name", "trust_post"),
        "trust_metric_value": float(result.get("trust_metric_value", 0.0) or 0.0),
        "trust_threshold": result.get("trust_threshold", 0.70),
        "trust_threshold_met": bool(result.get("trust_threshold_met", False)),
        "trust_post": float(result.get("trust_post", 0.0) or 0.0),
        "coverage": result.get("coverage"),
        "diversity": result.get("diversity"),
        "num_subclaims": result.get("num_subclaims"),
        "used_web_search": bool(result.get("used_web_search", False)),
        "data_source": result.get("data_source", "cache"),
        "degraded_mode": bool(llm_meta.get("degraded_mode", False)),
        "llm": llm_meta,
        "evidence": [
            {
                "statement": e.get("statement", ""),
                "source_url": e.get("source_url", ""),
                "score": round(float(e.get("final_score", 0) or 0), 3),
                "sem_score": round(float(e.get("sem_score", 0.0) or 0.0), 3),
                "kg_score": round(float(e.get("kg_score", 0.0) or 0.0), 3),
                "credibility": e.get("credibility"),
                "grade": e.get("grade", "N/A"),
            }
            for e in ranked_evidence[:5]
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker_service": SERVICE_NAME,
    }


def _format_fallback_response(payload: VerifyRequest, error_message: str) -> dict[str, Any]:
    verdict, confidence, truthfulness, rationale = _mock_verdict_for_claim(payload.claim)
    return {
        "status": "completed",
        "job_id": payload.job_id,
        "room_id": payload.room_id,
        "client_id": payload.client_id,
        "client_claim_id": payload.client_claim_id,
        "claim": payload.claim,
        "pipeline_status": "fallback",
        "result_status": "fallback",
        "verdict": verdict,
        "verdict_confidence": confidence,
        "truthfulness_percent": truthfulness,
        "verdict_rationale": f"{rationale} (fallback reason: {error_message})",
        "key_findings": ["Fallback mode used due to pipeline error."],
        "claim_breakdown": [],
        "evidence_map": [],
        "evidence": [],
        "evidence_count": 0,
        "facts_extracted": 0,
        "semantic_candidates_count": 0,
        "kg_candidates_count": 0,
        "vdb_signal_count": 0,
        "kg_signal_count": 0,
        "vdb_signal_sum_top5": 0.0,
        "kg_signal_sum_top5": 0.0,
        "top_ranking_score": 0.0,
        "avg_ranking_score": 0.0,
        "trust_policy_mode": "fixed",
        "trust_metric_name": "trust_post",
        "trust_metric_value": 0.0,
        "trust_threshold": 0.70,
        "trust_threshold_met": False,
        "trust_post": 0.0,
        "coverage": None,
        "diversity": None,
        "num_subclaims": None,
        "used_web_search": False,
        "data_source": "fallback",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker_service": SERVICE_NAME,
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/worker/test")
async def worker_test() -> dict[str, object]:
    rag_jobs_completed_total.inc()
    return {"status": "ok", "service": SERVICE_NAME, "message": "worker completion counter incremented"}


@app.post("/worker/verify")
async def worker_verify(payload: VerifyRequest) -> dict[str, object]:
    try:
        pipeline = await _get_pipeline()
        result = await pipeline.run(
            post_text=payload.claim,
            domain=payload.domain,
            round_id=payload.job_id,
            top_k=payload.top_k,
        )
        response = _format_completed_response(payload, result)
        rag_jobs_completed_total.inc()
        return response
    except Exception as exc:
        rag_jobs_failed_total.inc()
        # Preserve API availability if the full pipeline fails at runtime.
        fallback = _format_fallback_response(payload, str(exc))
        rag_jobs_completed_total.inc()
        return fallback


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "status": "running"}
