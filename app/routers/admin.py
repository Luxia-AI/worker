"""
Admin API routes for log management and monitoring.

Endpoints:
  GET /admin/logs - Retrieve logs with filters
  GET /admin/logs/{request_id} - Get all logs for a specific request
  GET /admin/logs/stats/{request_id} - Get log statistics
  WS  /admin/logs/stream - WebSocket for realtime log streaming
"""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.core.logger import get_logger
from app.services.logging.log_manager import LogManager

logger = get_logger(__name__)

router = APIRouter()

# Global reference to LogManager (set on app startup)
_log_manager: Optional[LogManager] = None


def set_log_manager(log_manager: LogManager) -> None:
    """Initialize the LogManager instance (called from main.py)."""
    global _log_manager
    _log_manager = log_manager


@router.get("/admin/logs", tags=["Admin", "Logs"])
async def get_logs(
    request_id: Optional[str] = Query(None, description="Filter by request/claim ID"),
    level: Optional[str] = Query(None, description="Filter by log level (INFO, WARNING, ERROR, DEBUG)"),
    module: Optional[str] = Query(None, description="Filter by module name"),
    start_time: Optional[str] = Query(None, description="ISO datetime start (inclusive)"),
    end_time: Optional[str] = Query(None, description="ISO datetime end (inclusive)"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """
    Retrieve logs from SQLite with optional filtering.

    Example:
        GET /admin/logs?request_id=claim-123&level=ERROR&limit=50

    Returns:
        {
            "count": 50,
            "offset": 0,
            "limit": 100,
            "filters": {...},
            "logs": [...]
        }
    """
    if not _log_manager:
        return JSONResponse({"error": "LogManager not initialized"}, status_code=503)

    try:
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None

        logs = await _log_manager.get_logs(
            request_id=request_id,
            level=level,
            module=module,
            start_time=start,
            end_time=end,
            limit=limit,
            offset=offset,
        )

        logger.info(f"[AdminAPI] Retrieved {len(logs)} logs, request_id={request_id}, level={level}")

        return {
            "count": len(logs),
            "offset": offset,
            "limit": limit,
            "filters": {
                "request_id": request_id,
                "level": level,
                "module": module,
                "start_time": start_time,
                "end_time": end_time,
            },
            "logs": logs,
        }
    except ValueError as e:
        return JSONResponse({"error": f"Invalid datetime format: {e}"}, status_code=400)
    except Exception as e:
        logger.error(f"[AdminAPI] Error retrieving logs: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/admin/logs/{request_id}", tags=["Admin", "Logs"], response_model=None)
async def get_logs_for_request(
    request_id: str,
    limit: int = Query(200, ge=1, le=1000),
    level: Optional[str] = Query(None),
):
    """
    Get all logs for a specific request/claim (convenience endpoint).

    Example:
        GET /admin/logs/claim-xyz-123
    """
    if not _log_manager:
        return JSONResponse({"error": "LogManager not initialized"}, status_code=503)

    try:
        logs = await _log_manager.get_logs(request_id=request_id, level=level, limit=limit)
        logger.info(f"[AdminAPI] Retrieved {len(logs)} logs for request {request_id}")
        return {
            "request_id": request_id,
            "count": len(logs),
            "level_filter": level,
            "logs": logs,
        }
    except Exception as e:
        logger.error(f"[AdminAPI] Error retrieving logs for {request_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/admin/logs/stats/{request_id}", tags=["Admin", "Logs"], response_model=None)
async def get_logs_stats(request_id: str):
    """
    Get log statistics for a specific request.

    Example:
        GET /admin/logs/stats/claim-xyz-123

    Returns:
        {
            "request_id": "claim-xyz-123",
            "total_logs": 42,
            "error_count": 2,
            "warning_count": 5,
            "info_count": 30,
            "debug_count": 5,
            "modules": ["app.services.corrective.pipeline", "app.services.ranking", ...]
        }
    """
    if not _log_manager:
        return JSONResponse({"error": "LogManager not initialized"}, status_code=503)

    try:
        stats = await _log_manager.get_stats(request_id=request_id)
        stats["request_id"] = request_id
        logger.info(f"[AdminAPI] Retrieved stats for request {request_id}")
        return stats
    except Exception as e:
        logger.error(f"[AdminAPI] Error getting stats for {request_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.websocket("/admin/logs/stream")
async def websocket_log_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for realtime log streaming from Redis pub/sub.

    Usage (client-side):
        const ws = new WebSocket('ws://localhost:9000/admin/logs/stream?channel=logs:all');
        ws.onmessage = (event) => {
            const log = JSON.parse(event.data);
            console.log(log);
        };

    Query Parameters:
        - channel: Redis channel to subscribe to (default: logs:all)
          Options: logs:all, logs:{request_id}, logs:{level}

    Receives logs as JSON objects:
        {
            "id": "uuid",
            "level": "INFO",
            "message": "...",
            "module": "app.services...",
            "timestamp": "2025-11-20T15:30:45.123456",
            "request_id": "claim-123",
            "round_id": "uuid",
            "session_id": null,
            "context": {}
        }
    """
    if not _log_manager:
        await websocket.close(code=1008, reason="LogManager not initialized")
        return

    await websocket.accept()

    # Get channel from query params, default to "logs:all"
    channel = websocket.query_params.get("channel", "logs:all")
    logger.info(f"[AdminAPI] WebSocket connected, subscribing to channel: {channel}")

    try:
        # Subscribe to Redis channel
        async def on_log(log_record: Dict[str, Any]) -> None:
            try:
                await websocket.send_json(log_record)
            except Exception as e:
                logger.debug(f"[AdminAPI] Failed to send log to WebSocket: {e}")
                raise

        await _log_manager.redis_broadcaster.subscribe(channel, on_log)

    except WebSocketDisconnect:
        logger.info(f"[AdminAPI] WebSocket disconnected from channel: {channel}")

    except Exception as e:
        logger.error(f"[AdminAPI] WebSocket error on channel {channel}: {e}")
        try:
            await websocket.close(code=1011, reason=f"Internal error: {e}")
        except Exception:  # nosec
            pass  # WebSocket may already be closed
