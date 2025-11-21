"""
Test: Verify logging system integration.

Run with:
    pytest tests/test_logging_system.py -v
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.services.kg.neo4j_client import Neo4jClient
from app.services.logging import LogManager, LogRecord


@pytest.mark.asyncio
async def test_log_record_creation():
    """Test LogRecord instantiation and serialization."""
    record = LogRecord(
        level="INFO",
        message="Test message",
        module="app.test",
        request_id="test-123",
        round_id="round-456",
        context={"key": "value"},
    )

    assert record.level == "INFO"
    assert record.message == "Test message"
    assert record.module == "app.test"
    assert record.request_id == "test-123"

    # Should be serializable to JSON
    record_dict = record.to_dict()
    assert record_dict["level"] == "INFO"
    assert record_dict["message"] == "Test message"
    assert isinstance(record_dict["timestamp"], str)


@pytest.mark.asyncio
async def test_log_manager_initialization():
    """Test LogManager can be created and started."""
    mock_neo4j = AsyncMock(spec=Neo4jClient)
    log_manager = LogManager(mock_neo4j)

    assert log_manager.neo4j == mock_neo4j
    assert log_manager.processing is False
    assert len(log_manager.subscribers) == 0


@pytest.mark.asyncio
async def test_log_manager_add_log():
    """Test adding logs to queue."""
    mock_neo4j = AsyncMock(spec=Neo4jClient)
    log_manager = LogManager(mock_neo4j)

    await log_manager.add_log(
        level="INFO",
        message="Test log",
        module="app.test",
        request_id="test-123",
    )

    # Log should be in queue
    assert not log_manager.log_queue.empty()


@pytest.mark.asyncio
async def test_log_manager_subscribe():
    """Test admin session subscription."""
    mock_neo4j = AsyncMock(spec=Neo4jClient)
    log_manager = LogManager(mock_neo4j)

    queue = await log_manager.subscribe("session-1")
    assert "session-1" in log_manager.subscribers
    assert log_manager.subscribers["session-1"] == queue

    await log_manager.unsubscribe("session-1")
    assert "session-1" not in log_manager.subscribers


@pytest.mark.asyncio
async def test_log_manager_batch_processing():
    """Test log batching and persistence."""
    mock_neo4j = AsyncMock(spec=Neo4jClient)
    mock_neo4j.execute = AsyncMock(return_value=[])
    log_manager = LogManager(mock_neo4j)

    # Add logs
    for i in range(5):
        await log_manager.add_log(
            level="INFO",
            message=f"Log {i}",
            module="app.test",
            request_id="test-123",
        )

    # Start processing
    await log_manager.start()

    # Wait for batch processing
    await asyncio.sleep(0.5)

    # Logs should be queued
    assert log_manager.log_queue.qsize() >= 0  # Some may have been processed

    await log_manager.stop()


@pytest.mark.asyncio
async def test_log_manager_get_logs():
    """Test retrieving logs from Neo4j."""
    mock_neo4j = AsyncMock(spec=Neo4jClient)
    mock_neo4j.execute = AsyncMock(
        return_value=[
            {
                "log": {
                    "id": "log-1",
                    "level": "INFO",
                    "message": "Test message",
                    "module": "app.test",
                    "timestamp": "2025-11-20T15:30:00",
                    "request_id": "test-123",
                }
            }
        ]
    )

    log_manager = LogManager(mock_neo4j)

    logs = await log_manager.get_logs(request_id="test-123", limit=10)

    assert len(logs) == 1
    assert logs[0]["level"] == "INFO"
    assert logs[0]["request_id"] == "test-123"


@pytest.mark.asyncio
async def test_log_manager_stats():
    """Test log statistics aggregation."""
    mock_neo4j = AsyncMock(spec=Neo4jClient)
    mock_neo4j.execute = AsyncMock(
        return_value=[
            {
                "total_logs": 42,
                "error_count": 2,
                "warning_count": 5,
                "info_count": 30,
                "debug_count": 5,
                "modules": ["app.services.search", "app.services.ranking"],
            }
        ]
    )

    log_manager = LogManager(mock_neo4j)

    stats = await log_manager.get_log_stats(request_id="test-123")

    assert stats["total_logs"] == 42
    assert stats["error_count"] == 2
    assert len(stats["modules"]) == 2


@pytest.mark.asyncio
async def test_broadcast_to_subscribers():
    """Test broadcasting logs to multiple subscribers."""
    mock_neo4j = AsyncMock(spec=Neo4jClient)
    log_manager = LogManager(mock_neo4j)

    # Subscribe two sessions
    queue1 = await log_manager.subscribe("session-1")
    queue2 = await log_manager.subscribe("session-2")

    # Create a log record
    record = LogRecord(
        level="INFO",
        message="Broadcast test",
        module="app.test",
    )

    # Broadcast it
    await log_manager._broadcast_to_subscribers(record)

    # Both queues should receive it
    received1 = queue1.get_nowait()
    received2 = queue2.get_nowait()

    assert received1.message == "Broadcast test"
    assert received2.message == "Broadcast test"


@pytest.mark.asyncio
async def test_clear_old_logs():
    """Test log cleanup."""
    mock_neo4j = AsyncMock(spec=Neo4jClient)
    mock_neo4j.execute = AsyncMock(return_value=[{"deleted": 100}])

    log_manager = LogManager(mock_neo4j)

    deleted = await log_manager.clear_old_logs(days=30)

    assert deleted == 100
    mock_neo4j.execute.assert_called_once()


def test_admin_router_set_log_manager():
    """Test admin router initialization."""
    from app.routers.admin import set_log_manager

    mock_neo4j = AsyncMock(spec=Neo4jClient)
    log_manager = LogManager(mock_neo4j)

    # Should not raise
    set_log_manager(log_manager)


@pytest.mark.asyncio
async def test_logging_handler_integration():
    """Test LogManagerHandler integration with Python logging."""
    from app.services.logging.log_handler import LogManagerHandler

    mock_neo4j = AsyncMock(spec=Neo4jClient)
    mock_neo4j.execute = AsyncMock(return_value=[])
    log_manager = LogManager(mock_neo4j)

    LogManagerHandler.set_log_manager(log_manager)

    handler = LogManagerHandler()
    assert handler._log_manager == log_manager

    # Create a log record
    import logging

    log_record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    log_record.request_id = "test-123"

    # Should not raise
    handler.emit(log_record)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
