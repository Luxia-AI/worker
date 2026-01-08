"""
Admin API routes for log management, monitoring, and domain trust management.

Endpoints:
  GET /admin/logs - Retrieve logs with filters
  GET /admin/logs/{request_id} - Get all logs for a specific request
  GET /admin/logs/stats/{request_id} - Get log statistics
  WS  /admin/logs/stream - WebSocket for realtime log streaming
  POST /admin/domains/approve - Approve a domain for trust
  POST /admin/domains/reject - Reject a domain
  GET /admin/domains/status - Get domain trust status
"""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.core.logger import get_logger
from app.services.domain_trust import get_domain_trust_store
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


# ============================================================================
# DOMAIN TRUST MANAGEMENT ENDPOINTS
# ============================================================================


@router.post("/admin/domains/approve", tags=["Admin", "Domain Trust"])
async def approve_domain(
    domain: str = Query(..., description="Domain to approve (e.g., example.com)"),
    reason: Optional[str] = Query(None, description="Optional reason for approval"),
    approved_by: str = Query("admin", description="Username of approving admin"),
):
    """
    Admin approves a domain for trust.

    When a domain is approved, evidence previously marked PENDING_DOMAIN_TRUST
    for that domain can be revalidated and their verdicts updated.

    Example:
        POST /admin/domains/approve?domain=example.com&approved_by=alice&reason=Verified+source

    Returns:
        {
            "status": "approved",
            "domain": "example.com",
            "approved_at": "2025-12-20T15:30:45.123456",
            "approved_by": "alice",
            "reason": "Verified source",
            "message": "Domain approved. Revalidation of pending evidence recommended."
        }
    """
    try:
        domain_trust = get_domain_trust_store()
        record = await domain_trust.approve_domain(domain, approved_by, reason)

        logger.info(f"[AdminAPI] Domain '{domain}' approved by {approved_by}")

        return {
            "status": "approved",
            "domain": domain,
            "approved_at": record.approved_at.isoformat(),
            "approved_by": record.approved_by,
            "reason": record.reason,
            "message": "Domain approved. Revalidation of pending evidence recommended.",
        }
    except Exception as e:
        logger.error(f"[AdminAPI] Error approving domain {domain}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/admin/domains/reject", tags=["Admin", "Domain Trust"])
async def reject_domain(
    domain: str = Query(..., description="Domain to reject"),
    reason: Optional[str] = Query(None, description="Optional reason for rejection"),
    approved_by: str = Query("admin", description="Username of approving admin"),
):
    """
    Admin rejects a domain for trust.

    When a domain is rejected, evidence from that domain is marked REVOKED
    and should not be trusted even if previously marked as PENDING_DOMAIN_TRUST.

    Example:
        POST /admin/domains/reject?domain=untrusted.com&approved_by=alice&reason=Misinformation+source

    Returns:
        {
            "status": "rejected",
            "domain": "untrusted.com",
            "rejected_at": "2025-12-20T15:30:45.123456",
            "rejected_by": "alice",
            "reason": "Misinformation source"
        }
    """
    try:
        domain_trust = get_domain_trust_store()
        record = await domain_trust.reject_domain(domain, approved_by, reason)

        logger.info(f"[AdminAPI] Domain '{domain}' rejected by {approved_by}")

        return {
            "status": "rejected",
            "domain": domain,
            "rejected_at": record.approved_at.isoformat(),
            "rejected_by": record.approved_by,
            "reason": record.reason,
        }
    except Exception as e:
        logger.error(f"[AdminAPI] Error rejecting domain {domain}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/admin/domains/status", tags=["Admin", "Domain Trust"])
async def get_domain_trust_status(
    domain: Optional[str] = Query(None, description="Specific domain to check (optional)"),
):
    """
    Get domain trust status.

    If domain is provided, returns status for that specific domain.
    Otherwise, returns aggregated trust status for all approved/rejected domains.

    Example:
        GET /admin/domains/status?domain=example.com
        GET /admin/domains/status  # Get all approved/rejected domains

    Returns:
        {
            "timestamp": "2025-12-20T15:30:45.123456",
            "domain_specific": {...} or null,
            "approved_count": 5,
            "rejected_count": 2,
            "approved_domains": ["example.com", "trusted.org", ...],
            "rejected_domains": ["bad.com", ...],
        }
    """
    try:
        domain_trust = get_domain_trust_store()

        response: dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "domain_specific": None,
            "approved_count": 0,
            "rejected_count": 0,
            "approved_domains": [],
            "rejected_domains": [],
        }

        if domain:
            # Get specific domain status
            record = domain_trust.get_record(domain)
            if record:
                response["domain_specific"] = {
                    "domain": domain,
                    "is_trusted": record.is_trusted,
                    "approved_at": record.approved_at.isoformat(),
                    "approved_by": record.approved_by,
                    "reason": record.reason,
                }
            else:
                response["domain_specific"] = {
                    "domain": domain,
                    "status": "no_admin_decision",
                    "message": "Domain has no admin approval/rejection; may be PENDING_DOMAIN_TRUST",
                }
        else:
            # Get aggregate status
            approved = domain_trust.get_approved_domains()
            rejected = domain_trust.get_rejected_domains()

            response["approved_count"] = len(approved)
            response["rejected_count"] = len(rejected)
            response["approved_domains"] = sorted(approved)
            response["rejected_domains"] = sorted(rejected)

        return response

    except Exception as e:
        logger.error(f"[AdminAPI] Error getting domain trust status: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
