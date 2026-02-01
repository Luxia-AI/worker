import asyncio
import json

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from fastapi import FastAPI

from app.core.config import settings
from app.core.logger import get_logger
from app.routers.admin import router as admin_router
from app.routers.admin import set_log_manager
from app.routers.pinecone import router as pinecone_router
from app.services.logging import LogManager
from app.services.logging.log_handler import LogManagerHandler

logger = get_logger(__name__)

# Global instances
_log_manager: LogManager | None = None
_kafka_consumer: AIOKafkaConsumer | None = None
_kafka_producer: AIOKafkaProducer | None = None


async def process_jobs():
    """Background task to process jobs from Kafka."""
    global _kafka_consumer, _kafka_producer
    if not _kafka_consumer or not _kafka_producer:
        logger.error("Kafka consumer or producer not initialized")
        return

    try:
        async for message in _kafka_consumer:
            job_data = message.value
            job_id = job_data.get("job_id")
            logger.info(f"Received job: {job_id}")

            # Send started update
            await _kafka_producer.send(
                "jobs.results",
                {
                    "job_id": job_id,
                    "status": "processing",
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )

            # Simulate processing (replace with actual processing)
            await asyncio.sleep(5)  # Simulate work

            # Send completed update
            await _kafka_producer.send(
                "jobs.results",
                {
                    "job_id": job_id,
                    "status": "completed",
                    "results": {"message": "Job completed"},
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )
            logger.info(f"Completed job: {job_id}")

    except Exception as e:
        logger.error(f"Error processing jobs: {e}")


async def startup_event() -> None:
    """Initialize logging system and Kafka consumer on app startup."""
    global _log_manager, _kafka_consumer, _kafka_producer

    try:
        # Create LogManager with Redis + SQLite
        # Redis for realtime streaming, SQLite for persistence
        _log_manager = LogManager(redis_url=settings.REDIS_URL, db_path=settings.LOG_DB_PATH)

        # Register with logging handler
        LogManagerHandler.set_log_manager(_log_manager)

        # Register with admin router
        set_log_manager(_log_manager)

        # Start background log processor
        await _log_manager.start()

        logger.info(f"[Main] LogManager initialized with Redis={settings.REDIS_URL}, DB={settings.LOG_DB_PATH}")

        # Get Kafka configuration (supports SASL/SSL for Azure Event Hubs)
        kafka_config = settings.get_kafka_config()

        # Initialize Kafka producer
        _kafka_producer = AIOKafkaProducer(
            **kafka_config,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await _kafka_producer.start()

        # Initialize Kafka consumer for jobs
        _kafka_consumer = AIOKafkaConsumer(
            "jobs.to_worker",  # Topic from dispatcher
            **kafka_config,
            group_id="worker-general-group",
            auto_offset_reset="latest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await _kafka_consumer.start()

        # Start background job processor
        asyncio.create_task(process_jobs())

        logger.info(f"[Main] Kafka producer and consumer started (bootstrap={settings.KAFKA_BOOTSTRAP})")

    except Exception as e:
        logger.error(f"[Main] Failed to initialize: {e}")


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

logger.info("Luxia Worker Service initialized")
