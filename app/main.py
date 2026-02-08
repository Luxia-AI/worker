import asyncio
import json
import time
import traceback

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import get_logger
from app.core.observability import (
    get_trace_context,
    metrics_payload,
    setup_tracing,
    stage_timer,
    worker_external_calls_total,
    worker_fallback_total,
    worker_job_duration_seconds,
    worker_jobs_completed_total,
    worker_jobs_consumed_total,
    worker_jobs_failed_total,
    worker_jobs_in_flight,
    worker_kafka_consume_failures_total,
    worker_kafka_send_failures_total,
    worker_queue_estimate,
)
from app.routers.admin import router as admin_router
from app.routers.admin import set_log_manager
from app.routers.pinecone import router as pinecone_router
from app.routers.scraper import router as scraper_router
from app.routers.scraper import set_scraper
from app.services.corrective.pipeline import CorrectivePipeline
from app.services.kg.neo4j_client import Neo4jClient
from app.services.logging import LogManager
from app.services.logging.log_handler import LogManagerHandler
from app.services.scraper import WHOScraper

logger = get_logger(__name__)

# Global instances
_log_manager: LogManager | None = None
_kafka_consumer: AIOKafkaConsumer | None = None
_kafka_producer: AIOKafkaProducer | None = None
_pipeline: CorrectivePipeline | None = None
_scraper: WHOScraper | None = None
_neo4j_client: Neo4jClient | None = None
_completed_emit_guard: dict[str, float] = {}
_COMPLETED_GUARD_TTL_SECONDS = 3600.0


def _gc_completed_emit_guard(now: float) -> None:
    expired = [job for job, ts in _completed_emit_guard.items() if now - ts > _COMPLETED_GUARD_TTL_SECONDS]
    for job in expired:
        _completed_emit_guard.pop(job, None)


async def emit_job_event(
    room_id: str | None,
    job_id: str | None,
    event_type: str,
    payload: dict,
) -> bool:
    """
    Unified job event emitter.
    event_type: stage | progress | completed | error
    Returns True if emitted, False when dropped by idempotency guard.
    """
    global _kafka_producer
    if not _kafka_producer or not job_id:
        return False

    now = time.time()
    _gc_completed_emit_guard(now)
    if event_type == "completed":
        if job_id in _completed_emit_guard:
            logger.warning("[Job %s] Completed event already emitted; dropping duplicate", job_id)
            return False
        _completed_emit_guard[job_id] = now

    event = {
        "event_type": event_type,
        "job_id": job_id,
        "room_id": room_id,
        "timestamp": now,
        **payload,
    }
    trace_meta = {k: v for k, v in get_trace_context().items() if v}
    if trace_meta:
        event["meta"] = {**(event.get("meta", {}) or {}), **trace_meta}
    try:
        await _kafka_producer.send("jobs.results", event)
        return True
    except Exception:
        worker_kafka_send_failures_total.inc()
        raise


async def process_jobs():
    """Background task to process jobs from Kafka."""
    global _kafka_consumer, _kafka_producer, _pipeline
    if not _kafka_consumer or not _kafka_producer:
        logger.error("Kafka consumer or producer not initialized")
        return

    # Initialize the RAG pipeline
    try:
        _pipeline = CorrectivePipeline()
        logger.info("[Main] CorrectivePipeline initialized")
    except Exception as e:
        logger.warning(f"[Main] CorrectivePipeline init failed (will use fallback): {e}")
        _pipeline = None

    try:
        async for message in _kafka_consumer:
            worker_jobs_consumed_total.inc()
            worker_jobs_in_flight.inc()
            job_data = message.value
            job_id = job_data.get("job_id")
            inbound_meta = job_data.get("meta", {}) or {}
            worker_queue_estimate.set(max((message.offset or 0), 0))

            # Extract post data from the job envelope (sent by dispatcher)
            post_data = job_data.get("post", {})
            post_id = post_data.get("post_id")
            room_id = post_data.get("room_id")
            claim_text = post_data.get("text", "")
            domain = job_data.get("assigned_worker_group", "general")

            logger.info(f"Received job: {job_id} (post={post_id}, room={room_id})")
            logger.info(f"Claim to verify: {claim_text[:100]}...")

            # Send started update (include post_id and room_id for socket routing)
            await emit_job_event(
                room_id,
                job_id,
                "stage",
                {
                    "post_id": post_id,
                    "status": "processing",
                    "job": {"stage": "started"},
                    "payload": {"message": "Starting RAG pipeline..."},
                },
            )
            stage_started_at = time.time()
            job_started_at = time.perf_counter()

            async def _emit_stage_event(stage: str, payload: dict | None = None) -> None:
                # completed is emitted only once from final result block below
                if stage == "completed":
                    return
                await emit_job_event(
                    room_id,
                    job_id,
                    "stage",
                    {
                        "post_id": post_id,
                        "status": "processing",
                        "job": {"stage": stage},
                        "stage_started_at": stage_started_at,
                        "stage_timestamp": time.time(),
                        "payload": payload or {},
                    },
                )

            # Run the actual RAG pipeline
            try:
                if _pipeline and claim_text:
                    logger.info(f"[Job {job_id}] Running CorrectivePipeline...")
                    with stage_timer("pipeline"):
                        result = await _pipeline.run(
                            post_text=claim_text,
                            domain=domain,
                            round_id=job_id,
                            top_k=5,
                            stage_callback=_emit_stage_event,
                        )

                    # Extract key results
                    ranked_evidence = result.get("ranked", [])
                    facts_count = len(result.get("facts", []))
                    status = result.get("status", "completed")
                    verdict_result = result.get("verdict", {})
                    llm_meta = result.get("llm", {})

                    # Build response with evidence and verdict
                    vdb_signal_count = sum(1 for e in ranked_evidence if float(e.get("sem_score", 0.0) or 0.0) > 0.0)
                    kg_signal_count = int(result.get("kg_signal_count", 0)) or sum(
                        1 for e in ranked_evidence if float(e.get("kg_score", 0.0) or 0.0) > 0.0
                    )
                    vdb_signal_sum = sum(float(e.get("sem_score", 0.0) or 0.0) for e in ranked_evidence[:5])
                    kg_signal_sum = float(result.get("kg_signal_sum_top5", 0.0)) or sum(
                        float(e.get("kg_score", 0.0) or 0.0) for e in ranked_evidence[:5]
                    )

                    response = {
                        "job_id": job_id,
                        "post_id": post_id,
                        "room_id": room_id,
                        "status": "completed",
                        "pipeline_status": "completed",
                        "result_status": status,
                        "claim": claim_text,
                        # Verdict (RAG Generation result)
                        "verdict": verdict_result.get("verdict", "UNVERIFIABLE"),
                        "verdict_confidence": verdict_result.get("confidence", 0.0),
                        "truthfulness_percent": verdict_result.get("truthfulness_percent", 0.0),
                        "verdict_rationale": verdict_result.get("rationale", ""),
                        "key_findings": verdict_result.get("key_findings", []),
                        "claim_breakdown": verdict_result.get("claim_breakdown", []),
                        # Evidence details
                        "evidence_count": len(ranked_evidence),
                        "facts_extracted": facts_count,
                        "semantic_candidates_count": result.get("semantic_candidates_count", 0),
                        "kg_candidates_count": result.get("kg_candidates_count", 0),
                        "vdb_signal_count": vdb_signal_count,
                        "kg_signal_count": kg_signal_count,
                        "vdb_signal_sum_top5": round(vdb_signal_sum, 3),
                        "kg_signal_sum_top5": round(kg_signal_sum, 3),
                        "debug_counts": result.get("debug_counts", {}),
                        "degraded_mode": bool(llm_meta.get("degraded_mode", False)),
                        "llm": llm_meta,
                        "evidence": [
                            {
                                "statement": e.get("statement", ""),
                                "source_url": e.get("source_url", ""),
                                "score": round(e.get("final_score", 0), 3),
                                "sem_score": round(float(e.get("sem_score", 0.0) or 0.0), 3),
                                "kg_score": round(float(e.get("kg_score", 0.0) or 0.0), 3),
                                "credibility": e.get("credibility"),
                                "grade": e.get("grade", "N/A"),
                            }
                            for e in ranked_evidence[:5]  # Top 5 evidence
                        ],
                        "evidence_map": verdict_result.get("evidence_map", []),
                        # Ranking metrics for debugging/display
                        "top_ranking_score": round(
                            result.get(
                                "ranking_top_score", ranked_evidence[0].get("final_score", 0) if ranked_evidence else 0
                            ),
                            3,
                        ),
                        "avg_ranking_score": round(
                            result.get(
                                "ranking_avg_score",
                                (
                                    (
                                        sum(float(e.get("final_score", 0) or 0.0) for e in ranked_evidence[:5])
                                        / max(1, len(ranked_evidence[:5]))
                                    )
                                    if ranked_evidence
                                    else 0.0
                                ),
                            ),
                            3,
                        ),
                        # Trust threshold info (VDB/KG cache vs external search)
                        "trust_threshold": result.get("trust_threshold", 0.70),
                        "trust_threshold_met": result.get("trust_threshold_met", False),
                        "used_web_search": result.get("used_web_search", False),
                        "data_source": result.get(
                            "data_source",
                            "WEB_SEARCH" if result.get("used_web_search", False) else "CACHE",
                        ),
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                    if inbound_meta:
                        response["meta"] = inbound_meta

                    logger.info(
                        f"[Job {job_id}] Pipeline completed: verdict={verdict_result.get('verdict', 'N/A')}, "
                        f"truthfulness={verdict_result.get('truthfulness_percent', 0)}%, "
                        f"confidence={verdict_result.get('confidence', 0):.2f}, "
                        f"{len(ranked_evidence)} evidence found"
                    )
                    worker_jobs_completed_total.inc()
                    worker_external_calls_total.labels(provider="pipeline", status="success").inc()

                else:
                    # Fallback if pipeline not available
                    logger.warning(f"[Job {job_id}] Pipeline not available, using fallback")
                    worker_fallback_total.inc()
                    response = {
                        "job_id": job_id,
                        "post_id": post_id,
                        "room_id": room_id,
                        "status": "completed",
                        "pipeline_status": "fallback",
                        "claim": claim_text,
                        "evidence_count": 0,
                        "message": "RAG pipeline not available - claim not verified",
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                    if inbound_meta:
                        response["meta"] = inbound_meta

            except Exception as e:
                # Log full traceback to identify exact source of error
                logger.error(f"[Job {job_id}] Pipeline error: {e}")
                logger.error(f"[Job {job_id}] Full traceback:\n{traceback.format_exc()}")
                worker_jobs_failed_total.inc()
                worker_external_calls_total.labels(provider="pipeline", status="error").inc()
                response = {
                    "job_id": job_id,
                    "post_id": post_id,
                    "room_id": room_id,
                    "status": "error",
                    "error": str(e),
                    "claim": claim_text,
                    "timestamp": asyncio.get_event_loop().time(),
                }
                if inbound_meta:
                    response["meta"] = inbound_meta

            # Send result with idempotent completion contract.
            # Treat terminal pipeline outcomes as completed for socket contract.
            terminal_result_statuses = {
                "completed",
                "completed_from_cache",
                "no_facts_extracted",
                "no_search_results",
                "no_queries_generated",
                "fallback",
            }
            result_status = str(
                response.get("result_status", response.get("pipeline_status", "completed")) or "completed"
            )
            if response.get("status") == "completed" and result_status in terminal_result_statuses:
                await emit_job_event(
                    room_id,
                    job_id,
                    "completed",
                    {
                        "post_id": post_id,
                        "status": "completed",
                        "pipeline_status": result_status,
                        "result": response,
                    },
                )
            elif response.get("status") == "completed":
                await emit_job_event(
                    room_id,
                    job_id,
                    "progress",
                    {
                        "post_id": post_id,
                        "status": "processing",
                        "pipeline_status": response.get("pipeline_status", "fallback"),
                        "result": response,
                    },
                )
            else:
                await emit_job_event(
                    room_id,
                    job_id,
                    "error",
                    {
                        "post_id": post_id,
                        "status": "error",
                        "result": response,
                    },
                )
            worker_job_duration_seconds.observe(time.perf_counter() - job_started_at)
            worker_jobs_in_flight.dec()
            logger.info(f"Completed job: {job_id}")

    except Exception as e:
        worker_kafka_consume_failures_total.inc()
        logger.error(f"Error processing jobs: {e}")


async def startup_event() -> None:
    """Initialize logging system and Kafka consumer on app startup."""
    global _log_manager, _kafka_consumer, _kafka_producer, _neo4j_client

    # Try to initialize LogManager (Redis + SQLite) - optional, pipeline works without it
    try:
        _log_manager = LogManager(redis_url=settings.REDIS_URL, db_path=settings.LOG_DB_PATH)
        LogManagerHandler.set_log_manager(_log_manager)
        set_log_manager(_log_manager)
        await _log_manager.start()
        logger.info(f"[Main] LogManager initialized with Redis={settings.REDIS_URL}, DB={settings.LOG_DB_PATH}")
    except Exception as e:
        logger.warning(f"[Main] LogManager failed (non-fatal, pipeline continues): {e}")
        _log_manager = None

    # Warm up Neo4j connection early to avoid first-use latency mid-pipeline
    try:
        _neo4j_client = Neo4jClient()
        start = time.monotonic()
        await _neo4j_client.execute("RETURN 1 AS ok")
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info("[Main] Neo4j connection pre-warmed in %.0f ms", elapsed_ms)
    except Exception as e:
        logger.warning(f"[Main] Neo4j pre-warm failed (non-fatal): {e}")
        _neo4j_client = None

    # Initialize Kafka - this is required for the pipeline
    try:
        kafka_config = settings.get_kafka_config()

        # Initialize Kafka producer
        _kafka_producer = AIOKafkaProducer(
            **kafka_config,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await _kafka_producer.start()
        logger.info(f"[Main] Kafka producer started (bootstrap={settings.KAFKA_BOOTSTRAP})")

        # Initialize Kafka consumer for jobs
        _kafka_consumer = AIOKafkaConsumer(
            "jobs.to_worker",  # Topic from dispatcher
            **kafka_config,
            group_id="worker-general-group",
            auto_offset_reset="latest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await _kafka_consumer.start()
        logger.info("[Main] Kafka consumer started (topic=jobs.to_worker)")

        # Start background job processor
        asyncio.create_task(process_jobs())
        logger.info("[Main] Job processor task started")

    except Exception as e:
        logger.error(f"[Main] Kafka initialization failed: {e}")

    # Initialize scraper
    global _scraper
    try:
        _scraper = WHOScraper()
        set_scraper(_scraper)
        logger.info("[Main] WHOScraper initialized")
    except Exception as e:
        logger.warning(f"[Main] WHOScraper initialization failed: {e}")
        _scraper = None


async def shutdown_event() -> None:
    """Cleanup on app shutdown."""
    if _log_manager:
        await _log_manager.stop()
        logger.info("[Main] LogManager stopped")

    if _kafka_consumer:
        await _kafka_consumer.stop()
        logger.info("[Main] Kafka consumer stopped")

    if _kafka_producer:
        await _kafka_producer.stop()
        logger.info("[Main] Kafka producer stopped")

    if _neo4j_client:
        await _neo4j_client.close()
        logger.info("[Main] Neo4j client closed")


app = FastAPI(title="Luxia Worker Service", version="1.0.0")
setup_tracing(app)

# Register startup/shutdown events
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# Include routers
app.include_router(pinecone_router, prefix="/worker", tags=["Pinecone"])
app.include_router(admin_router, tags=["Admin"])
app.include_router(scraper_router, prefix="/scraper", tags=["Scraper"])


# HTTP endpoint for direct claim verification (fallback when Kafka is unavailable)


class ClaimRequest(BaseModel):
    claim: str
    post_id: str | None = None
    room_id: str | None = None
    domain: str = "general"


@app.post("/worker/verify", tags=["Pipeline"])
async def verify_claim(request: ClaimRequest):
    """
    Verify a claim directly via HTTP (used when Kafka is unavailable).
    Returns the full verdict result.
    """
    global _pipeline

    # Initialize pipeline if not already done
    if _pipeline is None:
        try:
            _pipeline = CorrectivePipeline()
            logger.info("[HTTP] CorrectivePipeline initialized for HTTP request")
        except Exception as e:
            logger.error(f"[HTTP] Failed to initialize pipeline: {e}")
            return {
                "status": "error",
                "error": f"Pipeline initialization failed: {e}",
                "claim": request.claim,
            }

    job_id = request.post_id or f"http-{asyncio.get_event_loop().time()}"

    try:
        logger.info(f"[HTTP] Processing claim: {request.claim[:100]}...")

        with stage_timer("http_verify"):
            result = await _pipeline.run(
                post_text=request.claim,
                domain=request.domain,
                round_id=job_id,
                top_k=5,
            )

        # Extract key results
        ranked_evidence = result.get("ranked", [])
        facts_count = len(result.get("facts", []))
        status = result.get("status", "completed")
        verdict_result = result.get("verdict", {})

        response = {
            "job_id": job_id,
            "post_id": request.post_id,
            "room_id": request.room_id,
            "status": "completed",
            "pipeline_status": status,
            "claim": request.claim,
            # Verdict (RAG Generation result)
            "verdict": verdict_result.get("verdict", "UNVERIFIABLE"),
            "verdict_confidence": verdict_result.get("confidence", 0.0),
            "truthfulness_percent": verdict_result.get("truthfulness_percent", 0.0),
            "verdict_rationale": verdict_result.get("rationale", ""),
            "key_findings": verdict_result.get("key_findings", []),
            "claim_breakdown": verdict_result.get("claim_breakdown", []),
            # Evidence details
            "evidence_count": len(ranked_evidence),
            "facts_extracted": facts_count,
            "semantic_candidates_count": result.get("semantic_candidates_count", 0),
            "kg_candidates_count": result.get("kg_candidates_count", 0),
            "vdb_signal_count": sum(1 for e in ranked_evidence if float(e.get("sem_score", 0.0) or 0.0) > 0.0),
            "kg_signal_count": sum(1 for e in ranked_evidence if float(e.get("kg_score", 0.0) or 0.0) > 0.0),
            "vdb_signal_sum_top5": round(sum(float(e.get("sem_score", 0.0) or 0.0) for e in ranked_evidence[:5]), 3),
            "kg_signal_sum_top5": round(sum(float(e.get("kg_score", 0.0) or 0.0) for e in ranked_evidence[:5]), 3),
            "evidence": [
                {
                    "statement": e.get("statement", ""),
                    "source_url": e.get("source_url", ""),
                    "score": round(e.get("final_score", 0), 3),
                    "sem_score": round(float(e.get("sem_score", 0.0) or 0.0), 3),
                    "kg_score": round(float(e.get("kg_score", 0.0) or 0.0), 3),
                    "credibility": e.get("credibility"),
                    "grade": e.get("grade", "N/A"),
                }
                for e in ranked_evidence[:5]
            ],
            "evidence_map": verdict_result.get("evidence_map", []),
            "top_score": round(ranked_evidence[0]["final_score"], 3) if ranked_evidence else 0,
            "top_ranking_score": round(
                result.get("ranking_top_score", ranked_evidence[0].get("final_score", 0) if ranked_evidence else 0), 3
            ),
            "avg_ranking_score": round(
                result.get(
                    "ranking_avg_score",
                    (
                        (
                            sum(float(e.get("final_score", 0) or 0.0) for e in ranked_evidence[:5])
                            / max(1, len(ranked_evidence[:5]))
                        )
                        if ranked_evidence
                        else 0.0
                    ),
                ),
                3,
            ),
            "used_web_search": result.get("used_web_search", False),
            "data_source": result.get("data_source", "WEB_SEARCH" if result.get("used_web_search", False) else "CACHE"),
        }

        logger.info(
            f"[HTTP] Completed: verdict={verdict_result.get('verdict', 'N/A')}, "
            f"truthfulness={verdict_result.get('truthfulness_percent', 0)}%, "
            f"confidence={verdict_result.get('confidence', 0):.2f}, "
            f"{len(ranked_evidence)} evidence found"
        )

        return response

    except Exception as e:
        worker_jobs_failed_total.inc()
        logger.error(f"[HTTP] Pipeline error: {e}")
        logger.error(f"[HTTP] Full traceback:\n{traceback.format_exc()}")
        return {
            "job_id": job_id,
            "post_id": request.post_id,
            "room_id": request.room_id,
            "status": "error",
            "error": str(e),
            "claim": request.claim,
        }


@app.get("/metrics", tags=["Observability"])
async def metrics():
    payload, content_type = metrics_payload()
    return Response(content=payload, media_type=content_type)


@app.get("/health", tags=["Observability"])
async def health():
    return {
        "status": "ok",
        "kafka_consumer": _kafka_consumer is not None,
        "kafka_producer": _kafka_producer is not None,
        "redis_url_configured": bool(settings.REDIS_URL),
        "neo4j_configured": bool(settings.NEO4J_URI),
    }
