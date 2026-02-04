import asyncio
import json
import traceback

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from fastapi import FastAPI
from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import get_logger
from app.routers.admin import router as admin_router
from app.routers.admin import set_log_manager
from app.routers.pinecone import router as pinecone_router
from app.services.corrective.pipeline import CorrectivePipeline
from app.services.logging import LogManager
from app.services.logging.log_handler import LogManagerHandler

logger = get_logger(__name__)

# Global instances
_log_manager: LogManager | None = None
_kafka_consumer: AIOKafkaConsumer | None = None
_kafka_producer: AIOKafkaProducer | None = None
_pipeline: CorrectivePipeline | None = None


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
            job_data = message.value
            job_id = job_data.get("job_id")

            # Extract post data from the job envelope (sent by dispatcher)
            post_data = job_data.get("post", {})
            post_id = post_data.get("post_id")
            room_id = post_data.get("room_id")
            claim_text = post_data.get("text", "")
            domain = job_data.get("assigned_worker_group", "general")

            logger.info(f"Received job: {job_id} (post={post_id}, room={room_id})")
            logger.info(f"Claim to verify: {claim_text[:100]}...")

            # Send started update (include post_id and room_id for socket routing)
            await _kafka_producer.send(
                "jobs.results",
                {
                    "job_id": job_id,
                    "post_id": post_id,
                    "room_id": room_id,
                    "status": "processing",
                    "message": "Starting RAG pipeline...",
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )

            # Run the actual RAG pipeline
            try:
                if _pipeline and claim_text:
                    logger.info(f"[Job {job_id}] Running CorrectivePipeline...")
                    result = await _pipeline.run(
                        post_text=claim_text,
                        domain=domain,
                        round_id=job_id,
                        top_k=5,
                    )

                    # Extract key results
                    ranked_evidence = result.get("ranked", [])
                    facts_count = len(result.get("facts", []))
                    status = result.get("status", "completed")
                    verdict_result = result.get("verdict", {})

                    # Build response with evidence and verdict
                    response = {
                        "job_id": job_id,
                        "post_id": post_id,
                        "room_id": room_id,
                        "status": "completed",
                        "pipeline_status": status,
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
                        "evidence": [
                            {
                                "statement": e.get("statement", ""),
                                "source_url": e.get("source_url", ""),
                                "score": round(e.get("final_score", 0), 3),
                                "credibility": e.get("credibility"),
                                "grade": e.get("grade", "N/A"),
                            }
                            for e in ranked_evidence[:5]  # Top 5 evidence
                        ],
                        "evidence_map": verdict_result.get("evidence_map", []),
                        # Ranking metrics for debugging/display
                        "top_ranking_score": round(result.get("initial_top_score", 0), 3),
                        "avg_ranking_score": round(result.get("initial_top_score", 0), 3),
                        # Trust threshold info (VDB/KG cache vs external search)
                        "trust_threshold": result.get("trust_threshold", 0.70),
                        "trust_threshold_met": result.get("trust_threshold_met", False),
                        "used_web_search": result.get("used_web_search", False),
                        "data_source": "cache" if not result.get("used_web_search", False) else "web_search",
                        "timestamp": asyncio.get_event_loop().time(),
                    }

                    logger.info(
                        f"[Job {job_id}] Pipeline completed: verdict={verdict_result.get('verdict', 'N/A')}, "
                        f"truthfulness={verdict_result.get('truthfulness_percent', 0)}%, "
                        f"confidence={verdict_result.get('confidence', 0):.2f}, "
                        f"{len(ranked_evidence)} evidence found"
                    )

                else:
                    # Fallback if pipeline not available
                    logger.warning(f"[Job {job_id}] Pipeline not available, using fallback")
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

            except Exception as e:
                # Log full traceback to identify exact source of error
                logger.error(f"[Job {job_id}] Pipeline error: {e}")
                logger.error(f"[Job {job_id}] Full traceback:\n{traceback.format_exc()}")
                response = {
                    "job_id": job_id,
                    "post_id": post_id,
                    "room_id": room_id,
                    "status": "error",
                    "error": str(e),
                    "claim": claim_text,
                    "timestamp": asyncio.get_event_loop().time(),
                }

            # Send result
            await _kafka_producer.send("jobs.results", response)
            logger.info(f"Completed job: {job_id}")

    except Exception as e:
        logger.error(f"Error processing jobs: {e}")


async def startup_event() -> None:
    """Initialize logging system and Kafka consumer on app startup."""
    global _log_manager, _kafka_consumer, _kafka_producer

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


app = FastAPI(title="Luxia Worker Service", version="1.0.0")

# Register startup/shutdown events
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# Include routers
app.include_router(pinecone_router, prefix="/worker", tags=["Pinecone"])
app.include_router(admin_router, tags=["Admin"])


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
            "evidence": [
                {
                    "statement": e.get("statement", ""),
                    "source_url": e.get("source_url", ""),
                    "score": round(e.get("final_score", 0), 3),
                    "credibility": e.get("credibility"),
                    "grade": e.get("grade", "N/A"),
                }
                for e in ranked_evidence[:5]
            ],
            "evidence_map": verdict_result.get("evidence_map", []),
            "top_score": round(ranked_evidence[0]["final_score"], 3) if ranked_evidence else 0,
        }

        logger.info(
            f"[HTTP] Completed: verdict={verdict_result.get('verdict', 'N/A')}, "
            f"truthfulness={verdict_result.get('truthfulness_percent', 0)}%, "
            f"confidence={verdict_result.get('confidence', 0):.2f}, "
            f"{len(ranked_evidence)} evidence found"
        )

        return response

    except Exception as e:
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


logger.info("Luxia Worker Service initialized")
