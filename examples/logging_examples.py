"""
Example: Using the new logging system in your application code.

This shows how to:
1. Log with context (request_id, round_id)
2. Access logs via REST API
3. Stream logs in realtime via WebSocket
"""

import asyncio

from app.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Example 1: Logging with context (in pipeline code)
# ============================================================================
async def verify_claim(claim: str, claim_id: str) -> dict:
    """
    Example pipeline function that logs with request_id context.
    """
    logger.info(f"Starting verification for claim: {claim}", extra={"request_id": claim_id})

    try:
        # Simulate search phase
        logger.info("Searching trusted domains...", extra={"request_id": claim_id})
        await asyncio.sleep(0.5)

        # Simulate extraction
        logger.info(f"Extracting facts from {5} sources", extra={"request_id": claim_id})

        # Simulate ranking
        logger.info("Ranking candidates with hybrid scoring", extra={"request_id": claim_id})

        result = {
            "claim_id": claim_id,
            "claim": claim,
            "confidence": 0.85,
            "evidence": ["source1", "source2"],
        }

        logger.info(f"Verification completed with confidence {0.85}", extra={"request_id": claim_id})
        return result

    except Exception as e:
        logger.error(f"Verification failed: {e}", extra={"request_id": claim_id})
        raise


# ============================================================================
# Example 2: Accessing logs via REST API (from client)
# ============================================================================
async def example_rest_api():
    """
    Show how to fetch logs via HTTP.
    """
    import httpx

    client = httpx.AsyncClient()

    # Get all logs for a specific claim
    response = await client.get("http://localhost:9000/admin/logs/claim-123")
    logs = response.json()
    print(f"Found {logs['count']} logs for claim-123")

    # Get logs with filters
    response = await client.get(
        "http://localhost:9000/admin/logs",
        params={
            "request_id": "claim-123",
            "level": "ERROR",
            "limit": 50,
        },
    )
    error_logs = response.json()
    print(f"Found {len(error_logs['logs'])} error logs")

    # Get statistics
    response = await client.get("http://localhost:9000/admin/logs/stats/claim-123")
    stats = response.json()
    print(f"Stats: {stats['total_logs']} logs, {stats['error_count']} errors")

    await client.aclose()


# ============================================================================
# Example 3: Streaming logs in realtime via WebSocket
# ============================================================================
async def example_websocket_stream():
    """
    Show how to receive logs in realtime via WebSocket.
    """
    import json

    import websockets

    uri = "ws://localhost:9000/admin/logs/stream"

    async with websockets.connect(uri) as websocket:
        print("Connected to log stream. Waiting for logs...")

        try:
            while True:
                message = await websocket.recv()
                log = json.loads(message)

                # Pretty print incoming logs
                timestamp = log["timestamp"]
                level = log["level"]
                message_text = log["message"]
                request_id = log.get("request_id", "N/A")

                print(f"[{timestamp}] [{level}] (req={request_id}) {message_text}")

        except KeyboardInterrupt:
            print("Disconnecting...")


# ============================================================================
# Example 4: Structured context in nested functions
# ============================================================================
async def pipeline_phase(phase_name: str, request_id: str) -> None:
    """Simulate a pipeline phase with logging context."""
    logger.info(f"Starting {phase_name}", extra={"request_id": request_id})

    for i in range(3):
        logger.debug(f"Processing item {i+1} in {phase_name}", extra={"request_id": request_id})
        await asyncio.sleep(0.1)

    logger.info(f"Completed {phase_name}", extra={"request_id": request_id})


async def full_pipeline(claim: str, claim_id: str) -> None:
    """
    Simulate full pipeline with multiple phases logged together.

    In real code, round_id would be generated at pipeline start:
        round_id = str(uuid.uuid4())
    """
    import uuid

    round_id = str(uuid.uuid4())

    logger.info(
        "Starting pipeline for claim",
        extra={"request_id": claim_id, "round_id": round_id},
    )

    # Phase 1: Search
    await pipeline_phase("Search", claim_id)

    # Phase 2: Extraction
    await pipeline_phase("Extraction", claim_id)

    # Phase 3: Ranking
    await pipeline_phase("Ranking", claim_id)

    logger.info(
        "Pipeline completed",
        extra={"request_id": claim_id, "round_id": round_id},
    )


# ============================================================================
# Example 5: Integration with LogRecord context
# ============================================================================
class RequestContext:
    """Thread-local storage for request context (optional pattern)."""

    _request_id: str | None = None
    _round_id: str | None = None

    @classmethod
    def set(cls, request_id: str, round_id: str = None):
        cls._request_id = request_id
        cls._round_id = round_id

    @classmethod
    def get(cls) -> tuple[str | None, str | None]:
        return cls._request_id, cls._round_id


def log_with_context(message: str, level: str = "INFO"):
    """Helper to automatically inject context."""
    request_id, round_id = RequestContext.get()
    extra = {"request_id": request_id}
    if round_id:
        extra["round_id"] = round_id

    getattr(logger, level.lower())(message, extra=extra)


# ============================================================================
# Run examples
# ============================================================================
if __name__ == "__main__":
    print("Example: Basic logging with context")
    asyncio.run(verify_claim("The Earth is round", "claim-123"))

    print("\n" + "=" * 80)
    print("To test REST API, run:")
    print("  curl 'http://localhost:9000/admin/logs?request_id=claim-123'")
    print("  curl 'http://localhost:9000/admin/logs/stats/claim-123'")

    print("\n" + "=" * 80)
    print("To test WebSocket, run:")
    print(
        "  python -c 'import asyncio; from examples.logging_examples "
        "import example_websocket_stream; asyncio.run(example_websocket_stream())'"
    )
