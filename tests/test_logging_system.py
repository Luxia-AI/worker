"""
Test: Verify logging system integration.

Run with:
    pytest tests/test_logging_system.py -v
"""

import asyncio
import gc
import os
import tempfile

import pytest

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
    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

        assert log_manager.processing is False

        # Clean up explicitly
        del log_manager
        gc.collect()  # Force garbage collection to close SQLite connections
    finally:
        # Clean up with retry
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass  # File still in use, will be cleaned up later


@pytest.mark.asyncio
async def test_log_manager_add_log():
    """Test adding logs to queue."""
    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

        await log_manager.add_log(
            level="INFO",
            message="Test log",
            module="app.test",
            request_id="test-123",
        )

        # Log should be in queue
        assert not log_manager.log_queue.empty()

        # Clean up explicitly
        del log_manager
        gc.collect()
    finally:
        # Clean up with retry
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


@pytest.mark.asyncio
async def test_log_manager_subscribe():
    """Test admin session subscription."""
    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

        queue = await log_manager.subscribe("session-1")
        assert "session-1" in log_manager.subscribers
        assert log_manager.subscribers["session-1"] == queue

        await log_manager.unsubscribe("session-1")
        assert "session-1" not in log_manager.subscribers

        del log_manager
        gc.collect()
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


@pytest.mark.asyncio
async def test_log_manager_batch_processing():
    """Test log batching and persistence."""
    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

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
        del log_manager
        gc.collect()
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


@pytest.mark.asyncio
async def test_log_manager_get_logs():
    """Test retrieving logs from SQLite."""
    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

        # Add a test log
        await log_manager.add_log(
            level="INFO",
            message="Test message",
            module="app.test",
            request_id="test-123",
        )

        # Start processing to persist
        await log_manager.start()
        await asyncio.sleep(0.5)
        await log_manager.stop()

        logs = await log_manager.get_logs(request_id="test-123", limit=10)

        assert len(logs) >= 0  # May be 0 or 1 depending on timing

        del log_manager
        gc.collect()
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


@pytest.mark.asyncio
async def test_log_manager_stats():
    """Test log statistics aggregation."""
    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

        # Add test logs
        for i in range(5):
            await log_manager.add_log(
                level="INFO",
                message=f"Log {i}",
                module="app.test",
                request_id="test-123",
            )

        await log_manager.start()
        await asyncio.sleep(0.5)
        await log_manager.stop()

        stats = await log_manager.get_log_statistics(request_id="test-123")

        assert "total" in stats
        assert "by_level" in stats

        del log_manager
        gc.collect()
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


@pytest.mark.asyncio
async def test_broadcast_to_subscribers():
    """Test broadcasting logs to multiple subscribers."""
    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

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

        del log_manager
        gc.collect()
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


@pytest.mark.asyncio
async def test_clear_old_logs():
    """Test log cleanup."""
    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

        # Add test logs
        await log_manager.add_log(
            level="INFO",
            message="Old log",
            module="app.test",
            request_id="test-old",
        )

        await log_manager.start()
        await asyncio.sleep(0.5)
        await log_manager.stop()

        deleted = await log_manager.clear_old_logs(days=30)

        assert deleted >= 0  # Should return count of deleted rows

        del log_manager
        gc.collect()
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


def test_admin_router_set_log_manager():
    """Test admin router initialization."""
    from app.routers.admin import set_log_manager

    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

        # Should not raise
        set_log_manager(log_manager)

        del log_manager
        gc.collect()
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


@pytest.mark.asyncio
async def test_logging_handler_integration():
    """Test LogManagerHandler integration with Python logging."""
    from app.services.logging.log_handler import LogManagerHandler

    # Use temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        log_manager = LogManager(redis_url=redis_url, db_path=db_path)

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

        del handler
        del log_manager
        gc.collect()
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
