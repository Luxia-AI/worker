import asyncio
import contextlib
import logging
import os
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx
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
SOCKETHUB_BASE_URL = os.getenv("SOCKETHUB_URL", "").strip().rstrip("/")
SOCKETHUB_STAGE_CALLBACK_URL = os.getenv(
    "SOCKETHUB_STAGE_CALLBACK_URL",
    (f"{SOCKETHUB_BASE_URL}/internal/dispatch-stage" if SOCKETHUB_BASE_URL else ""),
).strip()
SOCKETHUB_RESULT_CALLBACK_TOKEN = os.getenv("SOCKETHUB_RESULT_CALLBACK_TOKEN", "").strip()

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
    domain: str = Field(default="health")
    top_k: int = Field(default=5, ge=1, le=20)


def _mock_verdict_for_claim(claim: str) -> tuple[str, float, float, str]:
    claim_l = claim.lower()
    if any(k in claim_l for k in ("hoax", "fake", "myth")):
        return ("FALSE", 0.86, 22.0, "Claim contains patterns commonly associated with misinformation framing.")
    if any(k in claim_l for k in ("significantly", "guaranteed", "always", "never", "detoxifies")):
        return ("FALSE", 0.64, 38.0, "Strong causal claim fallback: defaulting to conservative false until supported.")
    if any(k in claim_l for k in ("improves", "supports", "is linked to", "can reduce")):
        return ("TRUE", 0.71, 63.0, "Claim appears directionally supported in fallback mode.")
    return ("FALSE", 0.58, 40.0, "Fallback mode: conservative binary classification.")


def _truthfulness_band(score: float) -> str:
    value = max(0.0, min(100.0, float(score or 0.0)))
    if value <= 24.0:
        return "LIKELY_FALSE"
    if value <= 44.0:
        return "MOSTLY_FALSE"
    if value <= 55.0:
        return "MIXED_OR_UNCLEAR"
    if value <= 74.0:
        return "MOSTLY_TRUE"
    return "LIKELY_TRUE"


def _trust_failed_band(score: float) -> str:
    value = max(0.0, min(100.0, float(score or 0.0)))
    if value <= 44.0:
        return "MOSTLY_FALSE"
    if value <= 55.0:
        return "MIXED_OR_UNCLEAR"
    return "MOSTLY_TRUE"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _host(url: Any) -> str:
    val = str(url or "").strip()
    if not val:
        return ""
    try:
        return str(urllib.parse.urlparse(val).netloc or "").strip().lower()
    except Exception:
        return ""


def _build_debug_block(
    result: dict[str, Any], verdict_result: dict[str, Any], final_evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    ranked = result.get("ranked", []) if isinstance(result.get("ranked"), list) else []
    evidence_map = verdict_result.get("evidence_map", []) if isinstance(verdict_result, dict) else []
    claim_breakdown = verdict_result.get("claim_breakdown", []) if isinstance(verdict_result, dict) else []
    support_count = 0
    refute_count = 0
    neutral_count = 0
    for row in evidence_map:
        if not isinstance(row, dict):
            continue
        rel = str(row.get("relevance") or "").strip().upper()
        if rel == "SUPPORTS":
            support_count += 1
        elif rel == "REFUTES":
            refute_count += 1
        else:
            neutral_count += 1

    alignment_values: list[float] = []
    for seg in claim_breakdown:
        if not isinstance(seg, dict):
            continue
        dbg = seg.get("alignment_debug")
        if not isinstance(dbg, dict):
            continue
        for key in ("final_score", "score", "predicate_match_score"):
            if dbg.get(key) is not None:
                try:
                    alignment_values.append(float(dbg.get(key)))
                except (TypeError, ValueError):
                    pass
                break
    alignment_score = float(sum(alignment_values) / len(alignment_values)) if alignment_values else 0.0

    sources = {
        _host(e.get("source_url") if isinstance(e, dict) else "")
        for e in (evidence_map if evidence_map else final_evidence)
        if isinstance(e, dict)
    }
    sources.discard("")

    policy_path = verdict_result.get("policy_trace")
    if not isinstance(policy_path, list):
        policy_path = []
    if not policy_path:
        guard_reasons = verdict_result.get("verdict_guard_reasons", [])
        if isinstance(guard_reasons, list) and guard_reasons:
            policy_path = [{"step": "guards", "reasons": guard_reasons}]

    ranking_top = []
    for r in ranked[:5]:
        if not isinstance(r, dict):
            continue
        ranking_top.append(
            {
                "final_score": round(float(r.get("final_score", 0.0) or 0.0), 4),
                "sem_score": round(float(r.get("sem_score", 0.0) or 0.0), 4),
                "kg_score": round(float(r.get("kg_score", 0.0) or 0.0), 4),
                "support_score": round(float(r.get("support_score", 0.0) or 0.0), 4),
                "contradict_score": round(float(r.get("contradict_score", 0.0) or 0.0), 4),
                "stance": str(r.get("stance") or "neutral"),
            }
        )

    return {
        "claim_entities": result.get("claim_entities", []),
        "claim_predicate": (
            (result.get("predicate_target") or {}).get("predicate")
            if isinstance(result.get("predicate_target"), dict)
            else ""
        ),
        "generated_queries": result.get("queries_planned", result.get("queries", [])),
        "vector_hits": int(result.get("semantic_candidates_count", 0) or 0),
        "kg_hits": int(result.get("kg_candidates_count", 0) or 0),
        "retrieval_scores": {
            "top": result.get("retrieval_scores_top", []),
            "avg_top5": round(float(result.get("ranking_avg_score", 0.0) or 0.0), 4),
        },
        "ranking_scores": {
            "top": ranking_top,
            "support_strength": round(
                float(
                    result.get(
                        "support_strength",
                        verdict_result.get("support_mass", 0.0) if isinstance(verdict_result, dict) else 0.0,
                    )
                    or 0.0
                ),
                4,
            ),
            "contradiction_strength": round(
                float(
                    result.get(
                        "contradiction_strength",
                        verdict_result.get("contradict_mass", 0.0) if isinstance(verdict_result, dict) else 0.0,
                    )
                    or 0.0
                ),
                4,
            ),
        },
        "evidence_stance_distribution": {
            "supports": int(support_count),
            "refutes": int(refute_count),
            "neutral": int(neutral_count),
        },
        "evidence_sources": sorted(sources),
        "alignment_score": round(float(alignment_score), 4),
        "verdict_policy_path": policy_path,
        "trust_gate": {
            "threshold_met": bool(
                result.get(
                    "trust_threshold_met",
                    verdict_result.get("trust_threshold_met", False) if isinstance(verdict_result, dict) else False,
                )
            ),
            "sufficiency_reason": str(
                result.get("trust_sufficiency_reason") or result.get("adaptive_sufficiency_reason") or ""
            ),
        },
        "stop_reason": (
            str((result.get("pipeline_diagnostics_v2") or {}).get("stop_reason", ""))
            if isinstance(result.get("pipeline_diagnostics_v2"), dict)
            else ""
        ),
        "query_budget": result.get(
            "query_budget",
            {"total": int(result.get("queries_total", 0) or 0), "used": int(result.get("search_api_calls", 0) or 0)},
        ),
    }


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
    verdict_evidence = verdict_result.get("evidence", []) if isinstance(verdict_result, dict) else []
    if not isinstance(verdict_evidence, list):
        verdict_evidence = []
    final_evidence = verdict_evidence if verdict_evidence else ranked_evidence
    llm_meta = result.get("llm", {}) or {}
    status = str(result.get("status", "completed") or "completed")
    binary_first = _env_flag("VERDICT_POLICY_V3_ENABLED", True)
    verdict_internal = str(verdict_result.get("verdict", "UNVERIFIABLE") or "UNVERIFIABLE")
    verdict_binary = str(verdict_result.get("verdict_binary") or "").upper()
    if verdict_binary not in {"TRUE", "FALSE"}:
        truthfulness_hint = float(verdict_result.get("truth_score_binary", 0.5) or 0.5)
        verdict_binary = "TRUE" if truthfulness_hint >= 0.5 else "FALSE"
    response_verdict = verdict_binary if binary_first else verdict_internal
    display_verdict = response_verdict if binary_first else verdict_result.get("display_verdict", verdict_internal)
    debug_block_enabled = _env_flag("VERDICT_DEBUG_BLOCK_ENABLED", True)
    debug_block = _build_debug_block(result, verdict_result, final_evidence) if debug_block_enabled else {}

    return {
        "status": "completed",
        "job_id": payload.job_id,
        "room_id": payload.room_id,
        "client_id": payload.client_id,
        "client_claim_id": payload.client_claim_id,
        "claim": payload.claim,
        "pipeline_status": status,
        "result_status": status,
        "verdict": response_verdict,
        "verdict_binary": verdict_binary,
        "verdict_internal": verdict_internal,
        "abstain_reason": verdict_result.get("abstain_reason", ""),
        "display_verdict": display_verdict,
        "verdict_band": verdict_result.get("verdict_band"),
        "confidence": verdict_result.get("confidence", 0.0),
        "verdict_confidence": verdict_result.get("confidence", 0.0),
        "calibrated_confidence": verdict_result.get("calibrated_confidence", verdict_result.get("confidence", 0.0)),
        "truthfulness_percent": verdict_result.get("truthfulness_percent", 0.0),
        "class_probs": verdict_result.get("class_probs"),
        "calibration_meta": verdict_result.get("calibration_meta"),
        "evidence_attribution": verdict_result.get("evidence_attribution"),
        "verdict_rationale": verdict_result.get("rationale", ""),
        "key_findings": verdict_result.get("key_findings", []),
        "claim_breakdown": verdict_result.get("claim_breakdown", []),
        "evidence_map": verdict_result.get("evidence_map", []),
        "evidence_count": int(verdict_result.get("evidence_count", len(final_evidence)) or len(final_evidence)),
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
        "domain": result.get("domain", "health"),
        "health_scope": result.get("health_scope"),
        "pipeline_diagnostics_v2": result.get("pipeline_diagnostics_v2"),
        "trust_snapshot_v2": result.get("trust_snapshot_v2"),
        "support_mass": verdict_result.get("support_mass"),
        "contradict_mass": verdict_result.get("contradict_mass", verdict_result.get("refute_mass")),
        "neutral_mass": verdict_result.get("neutral_mass"),
        "sufficiency_score": verdict_result.get("sufficiency_score", verdict_result.get("evidence_sufficiency")),
        "policy_trace": verdict_result.get("policy_trace", []),
        "degraded_mode": bool(llm_meta.get("degraded_mode", False)),
        "llm": llm_meta,
        "evidence": [
            {
                "statement": e.get("statement", ""),
                "source_url": e.get("source_url", ""),
                "score": round(float(e.get("score", e.get("final_score", 0)) or 0), 3),
                "sem_score": round(float(e.get("sem_score", e.get("score", 0.0)) or 0.0), 3),
                "kg_score": round(float(e.get("kg_score", 0.0) or 0.0), 3),
                "credibility": e.get("credibility", 0.5),
                "grade": e.get("grade", "N/A"),
            }
            for e in final_evidence[:5]
        ],
        "debug": debug_block,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker_service": SERVICE_NAME,
    }


def _format_fallback_response(payload: VerifyRequest, error_message: str) -> dict[str, Any]:
    verdict, confidence, truthfulness, rationale = _mock_verdict_for_claim(payload.claim)
    verdict_band = _truthfulness_band(truthfulness)
    display_verdict = _trust_failed_band(truthfulness)
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
        "verdict_binary": verdict,
        "verdict_internal": "UNVERIFIABLE",
        "abstain_reason": "pipeline_fallback_error",
        "display_verdict": display_verdict,
        "verdict_band": verdict_band,
        "confidence": confidence,
        "verdict_confidence": confidence,
        "calibrated_confidence": confidence,
        "truthfulness_percent": truthfulness,
        "class_probs": None,
        "calibration_meta": None,
        "evidence_attribution": [],
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
        "domain": "health",
        "health_scope": None,
        "pipeline_diagnostics_v2": None,
        "trust_snapshot_v2": None,
        "support_mass": 0.0,
        "contradict_mass": 0.0,
        "neutral_mass": 0.0,
        "sufficiency_score": 0.0,
        "policy_trace": [{"step": "fallback", "reason": error_message}],
        "debug": {
            "claim_entities": [],
            "claim_predicate": "",
            "generated_queries": [],
            "vector_hits": 0,
            "kg_hits": 0,
            "retrieval_scores": {"top": [], "avg_top5": 0.0},
            "ranking_scores": {"top": [], "support_strength": 0.0, "contradiction_strength": 0.0},
            "evidence_stance_distribution": {"supports": 0, "refutes": 0, "neutral": 0},
            "evidence_sources": [],
            "alignment_score": 0.0,
            "verdict_policy_path": [{"step": "fallback"}],
            "trust_gate": {"threshold_met": False, "sufficiency_reason": "pipeline_fallback_error"},
            "stop_reason": "fallback",
            "query_budget": {"total": 0, "used": 0},
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker_service": SERVICE_NAME,
    }


def _stage_payload_dict(stage_payload: Any) -> dict[str, Any]:
    if isinstance(stage_payload, dict):
        return stage_payload
    return {"value": str(stage_payload)}


async def _emit_stage_callback(
    payload: VerifyRequest,
    stage: str,
    stage_payload: dict[str, Any],
) -> None:
    if not SOCKETHUB_STAGE_CALLBACK_URL:
        return

    body = {
        "event_type": "stage",
        "job_id": payload.job_id,
        "room_id": payload.room_id,
        "client_id": payload.client_id,
        "client_claim_id": payload.client_claim_id,
        "claim": payload.claim,
        "stage": stage,
        "stage_payload": stage_payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": SERVICE_NAME,
    }
    headers: dict[str, str] = {}
    if SOCKETHUB_RESULT_CALLBACK_TOKEN:
        headers["x-dispatcher-token"] = SOCKETHUB_RESULT_CALLBACK_TOKEN

    try:
        timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                SOCKETHUB_STAGE_CALLBACK_URL,
                json=body,
                headers=headers,
            )
            response.raise_for_status()
    except Exception as exc:
        logger.warning(
            "[Worker] Stage callback failed job_id=%s stage=%s room_id=%s error=%s",
            payload.job_id,
            stage,
            str(payload.room_id or ""),
            exc,
        )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/worker/test")
async def worker_test() -> dict[str, object]:
    rag_jobs_completed_total.inc()
    return {"status": "ok", "service": SERVICE_NAME, "message": "worker completion counter incremented"}


@app.post("/worker/verify")
async def worker_verify(payload: VerifyRequest) -> dict[str, object]:
    stage_events: list[dict[str, Any]] = []

    async def stage_callback(stage: str, callback_payload: Any) -> None:
        stage_payload = _stage_payload_dict(callback_payload)
        event = {
            "stage": stage,
            "payload": stage_payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        stage_events.append(event)
        await _emit_stage_callback(payload, stage=stage, stage_payload=stage_payload)

    try:
        pipeline = await _get_pipeline()
        result = await pipeline.run(
            post_text=payload.claim,
            domain=payload.domain,
            round_id=payload.job_id,
            top_k=payload.top_k,
            stage_callback=stage_callback,
        )
        response = _format_completed_response(payload, result)
        if stage_events:
            response["stage_events"] = stage_events
        rag_jobs_completed_total.inc()
        return response
    except Exception as exc:
        rag_jobs_failed_total.inc()
        with contextlib.suppress(Exception):
            await _emit_stage_callback(
                payload,
                stage="error",
                stage_payload={"error": str(exc)},
            )
        # Preserve API availability if the full pipeline fails at runtime.
        fallback = _format_fallback_response(payload, str(exc))
        if stage_events:
            fallback["stage_events"] = stage_events
        rag_jobs_completed_total.inc()
        return fallback


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "status": "running"}
