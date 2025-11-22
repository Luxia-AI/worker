"""
Custom logging handler to pipe logs to LogManager for persistence and realtime streaming.
"""

import logging
from typing import Optional

from app.services.logging.log_manager import LogManager


class LogManagerHandler(logging.Handler):
    """
    Custom logging handler that sends logs to LogManager.

    This allows all logs across the application to be:
    1. Persisted to Neo4j
    2. Streamed to admin clients via WebSocket
    3. Maintained with structured context (request_id, round_id, etc.)
    """

    # Class variable: shared LogManager instance (set by app startup)
    _log_manager: Optional[LogManager] = None

    def __init__(self):
        super().__init__()
        self.setLevel(logging.DEBUG)

    @classmethod
    def set_log_manager(cls, log_manager: LogManager) -> None:
        """Set the shared LogManager instance."""
        cls._log_manager = log_manager

    def emit(self, record: logging.LogRecord) -> None:
        """
        Send log record to LogManager for async processing.

        Called by logging system whenever a log is emitted.
        """
        if not self._log_manager:
            return

        try:
            # Extract context from record's extra fields (if provided)
            request_id = getattr(record, "request_id", None)
            round_id = getattr(record, "round_id", None)
            session_id = getattr(record, "session_id", None)
            context = getattr(record, "context", None)

            # Schedule async ingestion (don't block)
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            asyncio.create_task(
                self._log_manager.add_log(
                    level=record.levelname,
                    message=record.getMessage(),
                    module=record.name,
                    request_id=request_id,
                    round_id=round_id,
                    session_id=session_id,
                    context=context,
                )
            )
        except Exception:  # nosec
            # Silently fail if LogManager not ready (avoid logging loops)
            pass
