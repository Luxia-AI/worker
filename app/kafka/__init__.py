"""
Kafka service initialization and lifecycle management.

Provides factory functions to create and manage Kafka consumer/producer
with proper async lifecycle handling.
"""

import asyncio
from typing import Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.core.config import settings
from app.core.logger import get_logger
from app.kafka.consumer import WorkerJobConsumer
from app.kafka.producer import ResultPublisher

logger = get_logger(__name__)

# Global instances
_consumer: Optional[AIOKafkaConsumer] = None
_producer: Optional[AIOKafkaProducer] = None
_job_consumer: Optional[WorkerJobConsumer] = None
_consumer_task: Optional[asyncio.Task] = None


async def init_kafka_services() -> tuple[WorkerJobConsumer, AIOKafkaProducer]:
    """
    Initialize Kafka consumer and producer for the worker.

    Returns:
        Tuple of (WorkerJobConsumer, AIOKafkaProducer)
    """
    global _consumer, _producer, _job_consumer

    try:
        # Create producer
        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP,
            client_id="luxia-worker-producer",
            acks="all",
            retries=3,
        )
        await _producer.start()
        logger.info(f"[KafkaInit] Producer started: {settings.KAFKA_BOOTSTRAP}")

        # Create consumer
        _consumer = AIOKafkaConsumer(
            settings.JOBS_TOPIC,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP,
            group_id=settings.WORKER_GROUP_ID,
            client_id="luxia-worker-consumer",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            session_timeout_ms=settings.CONSUMER_TIMEOUT_MS,
        )
        await _consumer.start()
        logger.info(f"[KafkaInit] Consumer started: group={settings.WORKER_GROUP_ID}, " f"topic={settings.JOBS_TOPIC}")

        # Create publisher and job consumer
        publisher = ResultPublisher(_producer)
        _job_consumer = WorkerJobConsumer(_consumer, publisher)

        logger.info("[KafkaInit] Kafka services initialized successfully")
        return _job_consumer, _producer

    except Exception as e:
        logger.error(f"[KafkaInit] Failed to initialize Kafka services: {e}")
        await cleanup_kafka_services()
        raise


async def start_consumer_loop():
    """Start the consumer loop as a background task."""
    global _consumer_task

    if _job_consumer is None:
        raise RuntimeError("Kafka services not initialized. Call init_kafka_services() first.")

    try:
        _consumer_task = asyncio.create_task(_job_consumer.start_loop())
        logger.info("[KafkaInit] Consumer loop started as background task")
    except Exception as e:
        logger.error(f"[KafkaInit] Failed to start consumer loop: {e}")
        raise


async def cleanup_kafka_services():
    """Clean up Kafka services and cancel consumer loop."""
    global _consumer, _producer, _job_consumer, _consumer_task  # noqa

    # Cancel consumer loop
    if _consumer_task:
        _consumer_task.cancel()
        try:
            await _consumer_task
        except asyncio.CancelledError:
            logger.info("[KafkaCleanup] Consumer loop cancelled")

    # Close consumer
    if _consumer:
        try:
            await _consumer.stop()
            logger.info("[KafkaCleanup] Consumer stopped")
        except Exception as e:
            logger.warning(f"[KafkaCleanup] Error stopping consumer: {e}")
        finally:
            _consumer = None

    # Close producer
    if _producer:
        try:
            await _producer.stop()
            logger.info("[KafkaCleanup] Producer stopped")
        except Exception as e:
            logger.warning(f"[KafkaCleanup] Error stopping producer: {e}")
        finally:
            _producer = None

    _job_consumer = None
    logger.info("[KafkaCleanup] Kafka services cleaned up")


def get_job_consumer() -> Optional[WorkerJobConsumer]:
    """Get the global job consumer instance."""
    return _job_consumer


def get_producer() -> Optional[AIOKafkaProducer]:
    """Get the global producer instance."""
    return _producer
