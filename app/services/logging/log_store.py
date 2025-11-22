"""
SQLite persistence layer for logs.
Stores logs in a local database for historical access.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# Create logs database in project root
DB_PATH = Path(__file__).parent.parent.parent / "logs.db"


class LogStore:
    """SQLite-based log storage for persistence."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and schema if not exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS logs (
                        id TEXT PRIMARY KEY,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        module TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        request_id TEXT,
                        round_id TEXT,
                        session_id TEXT,
                        context TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                # Create indexes for fast queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON logs(request_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON logs(level)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_module ON logs(module)")

                conn.commit()
            logger.info(f"[LogStore] Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"[LogStore] Failed to initialize database: {e}")
            raise

    async def insert(self, log_record: Dict[str, Any]) -> None:
        """
        Insert a log record into SQLite.

        Args:
            log_record: Dict with id, level, message, module, timestamp, request_id, round_id, context
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO logs
                    (id, level, message, module, timestamp, request_id, round_id, session_id, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_record["id"],
                        log_record["level"],
                        log_record["message"],
                        log_record["module"],
                        log_record["timestamp"],
                        log_record.get("request_id"),
                        log_record.get("round_id"),
                        log_record.get("session_id"),
                        json.dumps(log_record.get("context", {})),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[LogStore] Failed to insert log: {e}")

    async def insert_batch(self, log_records: List[Dict[str, Any]]) -> None:
        """
        Batch insert multiple log records (more efficient).

        Args:
            log_records: List of log record dicts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO logs
                    (id, level, message, module, timestamp, request_id, round_id, session_id, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            record["id"],
                            record["level"],
                            record["message"],
                            record["module"],
                            record["timestamp"],
                            record.get("request_id"),
                            record.get("round_id"),
                            record.get("session_id"),
                            json.dumps(record.get("context", {})),
                        )
                        for record in log_records
                    ],
                )
                conn.commit()
            logger.debug(f"[LogStore] Inserted batch of {len(log_records)} logs")
        except Exception as e:
            logger.error(f"[LogStore] Failed to insert batch: {e}")

    async def query(
        self,
        request_id: Optional[str] = None,
        level: Optional[str] = None,
        module: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query logs with optional filters.

        Args:
            request_id: Filter by request ID
            level: Filter by log level
            module: Filter by module name
            start_time: Filter by start timestamp (ISO format)
            end_time: Filter by end timestamp (ISO format)
            limit: Results per page
            offset: Pagination offset

        Returns:
            List of log records as dicts
        """
        try:
            conditions = []
            params = []

            if request_id:
                conditions.append("request_id = ?")
                params.append(request_id)
            if level:
                conditions.append("level = ?")
                params.append(level)
            if module:
                conditions.append("module = ?")
                params.append(module)
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query_sql = f"""
                SELECT id, level, message, module, timestamp, request_id, round_id, session_id, context
                FROM logs
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """  # nosec B608
            params.extend([limit, offset])  # type: ignore[list-item]

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query_sql, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    results.append(
                        {
                            "id": row["id"],
                            "level": row["level"],
                            "message": row["message"],
                            "module": row["module"],
                            "timestamp": row["timestamp"],
                            "request_id": row["request_id"],
                            "round_id": row["round_id"],
                            "session_id": row["session_id"],
                            "context": json.loads(row["context"] or "{}"),
                        }
                    )

                return results
        except Exception as e:
            logger.error(f"[LogStore] Failed to query logs: {e}")
            return []

    async def get_stats(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for logs.

        Args:
            request_id: Filter by request ID

        Returns:
            Dict with total_logs, error_count, warning_count, etc.
        """
        try:
            where_clause = "WHERE request_id = ?" if request_id else "WHERE 1=1"
            params = [request_id] if request_id else []

            with sqlite3.connect(self.db_path) as conn:
                query_sql = f"""
                    SELECT
                        COUNT(*) as total_logs,
                        SUM(CASE WHEN level = 'ERROR' THEN 1 ELSE 0 END) as error_count,
                        SUM(CASE WHEN level = 'WARNING' THEN 1 ELSE 0 END) as warning_count,
                        SUM(CASE WHEN level = 'INFO' THEN 1 ELSE 0 END) as info_count,
                        SUM(CASE WHEN level = 'DEBUG' THEN 1 ELSE 0 END) as debug_count
                    FROM logs
                    {where_clause}
                """  # nosec B608
                cursor = conn.execute(query_sql, params)
                row = cursor.fetchone()

                return {
                    "total_logs": row[0] or 0,
                    "error_count": row[1] or 0,
                    "warning_count": row[2] or 0,
                    "info_count": row[3] or 0,
                    "debug_count": row[4] or 0,
                }
        except Exception as e:
            logger.error(f"[LogStore] Failed to get stats: {e}")
            return {
                "total_logs": 0,
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "debug_count": 0,
            }

    async def get_statistics(
        self,
        request_id: Optional[str] = None,
        level: Optional[str] = None,
        module: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed statistics about logs.

        Args:
            request_id: Filter by request ID
            level: Filter by log level
            module: Filter by module

        Returns:
            Statistics dict with total, by_level, by_module counts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Build WHERE clause
                where_parts = []
                params = []

                if request_id:
                    where_parts.append("request_id = ?")
                    params.append(request_id)
                if level:
                    where_parts.append("level = ?")
                    params.append(level)
                if module:
                    where_parts.append("module = ?")
                    params.append(module)

                where_clause = " AND ".join(where_parts) if where_parts else "1=1"

                # Get total count
                query_total = f"SELECT COUNT(*) as total FROM logs WHERE {where_clause}"  # nosec B608
                cursor = conn.execute(query_total, params)
                total = cursor.fetchone()["total"]

                # Get counts by level
                query_by_level = f"""
                    SELECT level, COUNT(*) as count
                    FROM logs
                    WHERE {where_clause}
                    GROUP BY level
                    """  # nosec B608
                cursor = conn.execute(query_by_level, params)
                by_level = {row["level"]: row["count"] for row in cursor.fetchall()}

                # Get counts by module
                query_by_module = f"""
                    SELECT module, COUNT(*) as count
                    FROM logs
                    WHERE {where_clause}
                    GROUP BY module
                    ORDER BY count DESC
                    LIMIT 20
                    """  # nosec B608
                cursor = conn.execute(query_by_module, params)
                by_module = {row["module"]: row["count"] for row in cursor.fetchall()}

                return {
                    "total": total,
                    "by_level": by_level,
                    "by_module": by_module,
                }
        except Exception as e:
            logger.error(f"[LogStore] Failed to get statistics: {e}")
            return {"total": 0, "by_level": {}, "by_module": {}}

    async def delete_old_logs(self, hours: int = 168) -> int:
        """
        Delete logs older than N hours (maintenance task).

        Args:
            hours: Delete logs older than this many hours (default: 7 days)

        Returns:
            Number of logs deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM logs
                    WHERE datetime(timestamp) < datetime('now', '-' || ? || ' hours')
                    """,
                    (hours,),
                )
                conn.commit()
                deleted = cursor.rowcount
                logger.info(f"[LogStore] Deleted {deleted} logs older than {hours} hours")
                return deleted
        except Exception as e:
            logger.error(f"[LogStore] Failed to delete old logs: {e}")
            return 0

    async def close(self) -> None:
        """Close database connection (if needed)."""
        logger.info("[LogStore] Database closed")
