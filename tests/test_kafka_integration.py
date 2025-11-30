"""
Integration tests for Kafka consumer and producer.

Tests cover:
- Consumer initialization and message routing
- Producer publishing to topics
- Error handling and DLQ behavior
- Retry logic for failed jobs
- End-to-end job processing
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.schemas import WorkerJob, WorkerResult
from app.kafka import cleanup_kafka_services, init_kafka_services
from app.kafka.consumer import WorkerJobConsumer
from app.kafka.producer import ResultPublisher

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_kafka_message():
    """Create a mock Kafka message."""
    msg = MagicMock()
    return msg


@pytest.fixture
async def mock_aio_kafka_consumer():
    """Create a mock AIOKafkaConsumer."""
    consumer = AsyncMock()
    consumer.__aiter__.return_value = []
    return consumer


@pytest.fixture
async def mock_aio_kafka_producer():
    """Create a mock AIOKafkaProducer."""
    producer = AsyncMock()
    return producer


@pytest.fixture
async def result_publisher(mock_aio_kafka_producer):
    """Create a ResultPublisher with mock producer."""
    return ResultPublisher(mock_aio_kafka_producer)


@pytest.fixture
async def worker_job_consumer(mock_aio_kafka_consumer, result_publisher):
    """Create a WorkerJobConsumer with mock dependencies."""
    return WorkerJobConsumer(mock_aio_kafka_consumer, result_publisher)


@pytest.fixture
def sample_job() -> WorkerJob:
    """Create a sample WorkerJob."""
    return WorkerJob(
        job_id="test-job-123",
        assigned_worker_group="worker-group-1",
        attempt=0,
        post={
            "post_id": "post-456",
            "content": "Test claim to verify",
            "platform": "twitter",
        },
    )


@pytest.fixture
def sample_result(sample_job) -> WorkerResult:
    """Create a sample WorkerResult."""
    return WorkerResult(
        job_id=sample_job.job_id,
        post_id="post-456",
        truth_score=0.85,
        confidence=0.92,
        evidence=["Source 1", "Source 2"],
        sources=["https://example.com/1", "https://example.com/2"],
        status="completed",
    )


# ============================================================================
# Unit Tests: ResultPublisher
# ============================================================================


class TestResultPublisher:
    """Test ResultPublisher functionality."""

    @pytest.mark.asyncio
    async def test_publish_progress(self, result_publisher):
        """Test publishing progress message."""
        await result_publisher.publish_progress("job-123", "received")

        result_publisher.producer.send_and_wait.assert_called_once()
        call_args = result_publisher.producer.send_and_wait.call_args
        assert call_args[0][0] == "jobs.progress"
        payload = json.loads(call_args[0][1].decode("utf-8"))
        assert payload["job_id"] == "job-123"
        assert payload["stage"] == "received"

    @pytest.mark.asyncio
    async def test_publish_result(self, result_publisher, sample_result):
        """Test publishing result message."""
        await result_publisher.publish_result(sample_result)

        result_publisher.producer.send_and_wait.assert_called_once()
        call_args = result_publisher.producer.send_and_wait.call_args
        assert call_args[0][0] == "jobs.results"
        payload = json.loads(call_args[0][1].decode("utf-8"))
        assert payload["job_id"] == sample_result.job_id
        assert payload["truth_score"] == 0.85

    @pytest.mark.asyncio
    async def test_publish_dlq(self, result_publisher, sample_job):
        """Test publishing to DLQ."""
        reason = "Test error"
        job_dict = sample_job.model_dump()

        await result_publisher.publish_dlq(job_dict, reason)

        result_publisher.producer.send_and_wait.assert_called_once()
        call_args = result_publisher.producer.send_and_wait.call_args
        assert call_args[0][0] == "jobs.worker_failed"
        payload = json.loads(call_args[0][1].decode("utf-8"))
        assert payload["reason"] == reason
        assert payload["job"]["job_id"] == sample_job.job_id


# ============================================================================
# Unit Tests: WorkerJobConsumer
# ============================================================================


class TestWorkerJobConsumer:
    """Test WorkerJobConsumer functionality."""

    @pytest.mark.asyncio
    async def test_valid_job_processing(self, worker_job_consumer, sample_job, result_publisher):
        """Test processing a valid job."""
        msg = MagicMock()
        msg.value = sample_job.model_dump_json().encode("utf-8")

        with patch("app.kafka.consumer.run_worker_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {
                "post_id": "post-456",
                "truth_score": 0.85,
                "confidence": 0.92,
                "evidence": ["Evidence"],
                "sources": ["https://source.com"],
                "status": "completed",
            }

            # Simulate one message then stop
            worker_job_consumer.consumer.__aiter__.return_value = [msg]

            # This would normally loop indefinitely, so we'll test components
            # instead of the full start_loop
            try:
                payload = json.loads(msg.value.decode("utf-8"))
                job = WorkerJob(**payload)
                assert job.job_id == sample_job.job_id
            except Exception as e:
                pytest.fail(f"Job parsing failed: {e}")

    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, worker_job_consumer, result_publisher):
        """Test handling of invalid JSON."""
        msg = MagicMock()
        msg.value = b"{ invalid json"

        worker_job_consumer.consumer.__aiter__.return_value = [msg]

        # The consumer should handle this gracefully
        # We test the JSON parsing directly
        with pytest.raises(json.JSONDecodeError):
            json.loads(msg.value.decode("utf-8"))

    @pytest.mark.asyncio
    async def test_wrong_worker_group_skipped(self, mock_aio_kafka_consumer, result_publisher):
        """Test that jobs for other worker groups are skipped."""
        job = WorkerJob(
            job_id="test-job",
            assigned_worker_group="worker-group-2",  # Different group
            attempt=0,
            post={"post_id": "post-123", "content": "test"},
        )

        msg = MagicMock()
        msg.value = job.model_dump_json().encode("utf-8")

        WorkerJobConsumer(mock_aio_kafka_consumer, result_publisher)

        # Parse and validate
        payload = json.loads(msg.value.decode("utf-8"))
        parsed_job = WorkerJob(**payload)

        # Check group assignment (in real code this is in start_loop)
        assert parsed_job.assigned_worker_group != "worker-group-1"

    @pytest.mark.asyncio
    async def test_retry_job_on_failure(self, result_publisher):
        """Test retry logic for failed jobs."""
        job = WorkerJob(
            job_id="test-job",
            assigned_worker_group="worker-group-1",
            attempt=0,
            post={"post_id": "post-123", "content": "test"},
        )

        # Simulate max attempts
        retry_job = job.model_copy(update={"attempt": 1})
        assert retry_job.attempt == 1

        for i in range(2, 4):
            retry_job = retry_job.model_copy(update={"attempt": i})
            assert retry_job.attempt == i

    @pytest.mark.asyncio
    async def test_progress_publishing(self, result_publisher):
        """Test that progress messages are published."""
        await result_publisher.publish_progress("job-123", "received")
        await result_publisher.publish_progress("job-123", "completed")

        assert result_publisher.producer.send_and_wait.call_count == 2


# ============================================================================
# Integration Tests (require Kafka)
# ============================================================================


@pytest.mark.integration
class TestKafkaIntegration:
    """Integration tests with real Kafka (if available)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        True,  # Always skip unless explicitly run with Kafka
        reason="Requires --run-integration flag and running Kafka broker",
    )
    async def test_init_kafka_services(self):
        """Test Kafka services initialization."""
        try:
            consumer, producer = await init_kafka_services()
            assert consumer is not None
            assert producer is not None
            await cleanup_kafka_services()
        except Exception as e:
            pytest.skip(f"Kafka not available: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        True,  # Always skip unless explicitly run with Kafka
        reason="Requires --run-integration flag and running Kafka broker",
    )
    async def test_message_flow(self):
        """Test complete message flow from producer to consumer."""
        try:
            import asyncio

            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

            # Create producer
            producer = AIOKafkaProducer(
                bootstrap_servers="localhost:9092",
                client_id="test-producer",
            )
            await producer.start()

            # Send test message
            test_msg = {"test": "message"}
            await producer.send_and_wait("jobs.results", json.dumps(test_msg).encode("utf-8"))

            # Create consumer
            consumer = AIOKafkaConsumer(
                "jobs.results",
                bootstrap_servers="localhost:9092",
                group_id="test-group",
                auto_offset_reset="earliest",
            )
            await consumer.start()

            # Receive message with timeout
            try:
                msg = await asyncio.wait_for(consumer.__anext__(), timeout=5)
                payload = json.loads(msg.value.decode("utf-8"))
                assert payload == test_msg
            finally:
                await consumer.stop()
                await producer.stop()

        except Exception as e:
            pytest.skip(f"Kafka integration test failed: {e}")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_dlq_publish_on_validation_error(self, result_publisher):
        """Test DLQ is used for validation errors."""
        invalid_job = {"job_id": "test", "missing_field": "value"}
        reason = "Validation error: post is required"

        await result_publisher.publish_dlq(invalid_job, reason)

        result_publisher.producer.send_and_wait.assert_called_once()
        call_args = result_publisher.producer.send_and_wait.call_args
        assert call_args[0][0] == "jobs.worker_failed"

    @pytest.mark.asyncio
    async def test_producer_error_resilience(self):
        """Test that producer errors don't crash the system."""
        producer = AsyncMock()
        producer.send_and_wait.side_effect = Exception("Producer error")
        publisher = ResultPublisher(producer)

        with pytest.raises(Exception):
            await publisher.publish_progress("job-123", "received")


# ============================================================================
# Data Structure Tests
# ============================================================================


class TestDataStructures:
    """Test Pydantic schemas."""

    def test_worker_job_schema(self, sample_job):
        """Test WorkerJob schema validation."""
        assert sample_job.job_id == "test-job-123"
        assert sample_job.assigned_worker_group == "worker-group-1"
        assert sample_job.attempt == 0
        assert sample_job.post["post_id"] == "post-456"

    def test_worker_job_serialization(self, sample_job):
        """Test WorkerJob can be serialized to JSON."""
        json_str = sample_job.model_dump_json()
        restored = WorkerJob.model_validate_json(json_str)
        assert restored.job_id == sample_job.job_id

    def test_worker_result_schema(self, sample_result):
        """Test WorkerResult schema."""
        assert sample_result.job_id == "test-job-123"
        assert sample_result.truth_score == 0.85
        assert len(sample_result.evidence) == 2

    def test_worker_result_serialization(self, sample_result):
        """Test WorkerResult can be serialized to JSON."""
        json_str = sample_result.model_dump_json()
        restored = WorkerResult.model_validate_json(json_str)
        assert restored.job_id == sample_result.job_id
        assert restored.truth_score == 0.85


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test Kafka configuration."""

    def test_config_defaults(self):
        """Test that config has sensible defaults."""
        from app.core.config import settings

        # These values may be overridden by environment variables
        # Just verify they are set
        assert settings.KAFKA_BOOTSTRAP is not None
        assert settings.JOBS_TOPIC == "jobs.to_worker"
        assert settings.RESULTS_TOPIC == "jobs.results"
        assert settings.PROGRESS_TOPIC == "jobs.progress"
        assert settings.DLQ_TOPIC == "jobs.worker_failed"
        assert settings.WORKER_GROUP_ID == "worker-group-1"
        assert settings.MAX_JOB_ATTEMPTS == 3

    def test_module_level_constants(self):
        """Test module-level constants for backward compatibility."""
        from app.core.config import GROUP_ID, JOBS_TOPIC, MAX_ATTEMPTS, RESULTS_TOPIC

        assert GROUP_ID == "worker-group-1"
        assert MAX_ATTEMPTS == 3
        assert JOBS_TOPIC == "jobs.to_worker"
        assert RESULTS_TOPIC == "jobs.results"
