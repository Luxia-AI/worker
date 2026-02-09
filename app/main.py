import os
from datetime import datetime, timezone

from fastapi import FastAPI
from prometheus_client import Counter
from pydantic import BaseModel
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


class VerifyRequest(BaseModel):
    job_id: str
    claim: str
    room_id: str | None = None
    source: str | None = None


app = FastAPI(title="Luxia Worker", version=SERVICE_VERSION)
install_metrics(app, service_name=SERVICE_NAME, version=SERVICE_VERSION, env=SERVICE_ENV)


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
        verdict, confidence, truthfulness, rationale = _mock_verdict_for_claim(payload.claim)
        rag_jobs_completed_total.inc()
        return {
            "status": "completed",
            "job_id": payload.job_id,
            "room_id": payload.room_id,
            "claim": payload.claim,
            "verdict": verdict,
            "verdict_confidence": confidence,
            "truthfulness_percent": truthfulness,
            "verdict_rationale": rationale,
            "key_findings": [
                "Automated worker compatibility flow active.",
                "Real retrieval/ranking can be attached behind this endpoint.",
            ],
            "claim_breakdown": [],
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
            "trust_threshold": 0.70,
            "trust_threshold_met": False,
            "used_web_search": False,
            "data_source": "cache",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker_service": SERVICE_NAME,
        }
    except Exception as exc:
        rag_jobs_failed_total.inc()
        return {
            "status": "error",
            "job_id": payload.job_id,
            "room_id": payload.room_id,
            "claim": payload.claim,
            "message": f"Worker verification failed: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "status": "running"}
