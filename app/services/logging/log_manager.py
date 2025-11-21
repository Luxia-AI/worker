"""LogManager: Centralized logging service for realtime admin console and persistent storage.

Architecture:
- Redis pub/sub for realtime streaming to WebSocket clients
- SQLite for persistent log storage
- Async queue-based log collection from all modules
- Structured log context (request_id, round_id, session_id)
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.services.logging.log_store import LogStore
from app.services.logging.redis_broadcaster import RedisLogBroadcaster

logger = get_logger(__name__)


class LogRecord:
    """Structured log record with metadata."""

    def __init__(
        self,
        level: str,
        message: str,
        module: str,
        timestamp: Optional[datetime] = None,
        request_id: Optional[str] = None,
        round_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.level = level
        self.message = message
        self.module = module
        self.timestamp = timestamp or datetime.utcnow()
        self.request_id = request_id
        self.round_id = round_id
        self.session_id = session_id
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "level": self.level,
            "message": self.message,
            "module": self.module,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "round_id": self.round_id,
            "session_id": self.session_id,
            "context": self.context,
        }


class LogManager:
    """
    Manages log collection, realtime streaming, and persistent storage.

    Features:
    - Async queue for log collection (non-blocking)
    - Redis pub/sub for realtime streaming to WebSocket clients
    - SQLite for persistent log storage
    - Batch processing for efficiency
    - Structured context tracking

    Usage:
        log_manager = LogManager(
            redis_url="redis://localhost:6379",
            db_path="logs.db"
        )
        await log_manager.start()

        # Add logs
        await log_manager.add_log(
            level="INFO",
            message="Processing...",
            module="app.services",
            request_id="claim-123"
        )

        # Retrieve logs
        logs = await log_manager.get_logs(request_id="claim-123")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db_path: str = "logs.db",
    ):
        self.redis_broadcaster = RedisLogBroadcaster(redis_url)
        self.log_store = LogStore(db_path)
        self.log_queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        self._batch_size = 10
        self._batch_timeout = 5  # seconds
        self.subscribers: Dict[str, asyncio.Queue] = {}  # session_id -> queue

    async def start(self) -> None:
        """Start the logging system."""
        try:
            # Connect to Redis
            await self.redis_broadcaster.connect()

            self.processing = True
            logger.info("[LogManager] Starting background log processor")
            asyncio.create_task(self._process_logs_batch())

        except Exception as e:
            logger.error(f"[LogManager] Failed to start: {e}")
            raise

    async def stop(self) -> None:
        """Stop the logging system."""
        self.processing = False
        await self.redis_broadcaster.disconnect()
        logger.info("[LogManager] Logging system stopped")

    async def add_log(
        self,
        level: str,
        message: str,
        module: str,
        request_id: Optional[str] = None,
        round_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a log record to the queue for async processing.

        Args:
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            message: Log message text
            module: Module name (__name__)
            request_id: Unique request/claim ID
            round_id: Pipeline round UUID
            session_id: Admin session ID
            context: Additional context dict
        """
        record = LogRecord(
            level=level,
            message=message,
            module=module,
            request_id=request_id,
            round_id=round_id,
            session_id=session_id,
            context=context or {},
        )
        await self.log_queue.put(record)

    async def _process_logs_batch(self) -> None:
        """Background task: collect logs and persist/broadcast in batches."""
        batch: List[LogRecord] = []
        last_flush = datetime.utcnow()

        while self.processing:
            try:
                timeout = max(0.1, self._batch_timeout - (datetime.utcnow() - last_flush).total_seconds())
                record = await asyncio.wait_for(self.log_queue.get(), timeout=timeout)
                batch.append(record)

                # Publish to Redis immediately for realtime
                await self._publish_to_redis(record)

                # Also broadcast to local subscribers
                await self._broadcast_to_subscribers(record)

                # Flush if batch full
                if len(batch) >= self._batch_size:
                    await self._persist_batch(batch)
                    batch = []
                    last_flush = datetime.utcnow()

            except asyncio.TimeoutError:
                # Timeout: flush if batch has items
                if batch:
                    await self._persist_batch(batch)
                    batch = []
                    last_flush = datetime.utcnow()

            except Exception as e:
                logger.error(f"[LogManager] Error processing log batch: {e}")

    async def _publish_to_redis(self, record: LogRecord) -> None:
        """Publish log to Redis for realtime streaming."""
        try:
            await self.redis_broadcaster.publish(record.to_dict())
        except Exception as e:
            logger.error(f"[LogManager] Failed to publish to Redis: {e}")

    async def _persist_batch(self, batch: List[LogRecord]) -> None:
        """Persist batch of logs to SQLite."""
        if not batch:
            return

        try:
            log_dicts = [record.to_dict() for record in batch]
            await self.log_store.insert_batch(log_dicts)
            logger.debug(f"[LogManager] Persisted {len(batch)} logs to SQLite")
        except Exception as e:
            logger.error(f"[LogManager] Failed to persist logs to SQLite: {e}")

    async def get_logs(
        self,
        request_id: Optional[str] = None,
        level: Optional[str] = None,
        module: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs from SQLite with optional filters.

        Args:
            request_id: Filter by request/claim ID
            level: Filter by log level
            module: Filter by module name
            start_time: Filter by timestamp >= start_time
            end_time: Filter by timestamp <= end_time
            limit: Max results to return
            offset: Pagination offset

        Returns:
            List of log records as dicts
        """
        try:
            start_str = start_time.isoformat() if start_time else None
            end_str = end_time.isoformat() if end_time else None

            logs = await self.log_store.query(
                request_id=request_id,
                level=level,
                module=module,
                start_time=start_str,
                end_time=end_str,
                limit=limit,
                offset=offset,
            )
            return logs
        except Exception as e:
            logger.error(f"[LogManager] Failed to retrieve logs: {e}")
            return []

    async def get_stats(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for logs.

        Args:
            request_id: Filter by request/claim ID

        Returns:
            Stats dict with total_logs, error_count, warning_count, etc.
        """
        try:
            stats = await self.log_store.get_stats(request_id=request_id)
            return stats
        except Exception as e:
            logger.error(f"[LogManager] Failed to get log stats: {e}")
            return {}

    async def clear_old_logs(self, days: int = 7) -> int:
        """
        Clean up logs older than N days (maintenance task).

        Args:
            days: Delete logs older than this many days (default: 7 days)

        Returns:
            Number of logs deleted
        """
        try:
            hours = days * 24
            deleted = await self.log_store.delete_old_logs(hours=hours)
            logger.info(f"[LogManager] Cleaned up {deleted} logs older than {days} days")
            return deleted
        except Exception as e:
            logger.error(f"[LogManager] Failed to clean up old logs: {e}")
            return 0

    async def get_active_subscribers(self) -> int:
        """Get count of active Redis subscribers (for monitoring)."""
        return await self.redis_broadcaster.get_active_channels()

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        """
        Subscribe to log stream for a specific session.

        Args:
            session_id: Unique session identifier

        Returns:
            Queue that will receive LogRecord objects
        """
        queue = asyncio.Queue()
        self.subscribers[session_id] = queue
        logger.info(f"[LogManager] Session {session_id} subscribed to log stream")
        return queue

    async def unsubscribe(self, session_id: str) -> None:
        """
        Unsubscribe from log stream.

        Args:
            session_id: Session identifier to unsubscribe
        """
        if session_id in self.subscribers:
            del self.subscribers[session_id]
            logger.info(f"[LogManager] Session {session_id} unsubscribed from log stream")

    async def _broadcast_to_subscribers(self, record: LogRecord) -> None:
        """
        Broadcast log record to all subscribed sessions.

        Args:
            record: LogRecord to broadcast
        """
        for session_id, queue in list(self.subscribers.items()):
            try:
                queue.put_nowait(record)
            except asyncio.QueueFull:
                logger.warning(f"[LogManager] Queue full for session {session_id}, dropping log")
            except Exception as e:
                logger.error(f"[LogManager] Failed to broadcast to session {session_id}: {e}")

    async def get_log_statistics(
        self,
        request_id: Optional[str] = None,
        level: Optional[str] = None,
        module: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics about logs.

        Args:
            request_id: Filter by request ID
            level: Filter by log level
            module: Filter by module

        Returns:
            Statistics dict with total, by_level, by_module counts
        """
        try:
            stats = await self.log_store.get_statistics(
                request_id=request_id,
                level=level,
                module=module,
            )
            return stats
        except Exception as e:
            logger.error(f"[LogManager] Failed to get log statistics: {e}")
            return {"total": 0, "by_level": {}, "by_module": {}}
