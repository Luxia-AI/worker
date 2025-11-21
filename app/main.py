from fastapi import FastAPI

from app.core.config import settings
from app.core.logger import get_logger
from app.routers.admin import router as admin_router
from app.routers.admin import set_log_manager
from app.routers.pinecone import router as pinecone_router
from app.services.logging import LogManager
from app.services.logging.log_handler import LogManagerHandler

logger = get_logger(__name__)

# Global LogManager instance
_log_manager: LogManager | None = None


async def startup_event() -> None:
    """Initialize logging system on app startup."""
    global _log_manager  # noqa: F841

    try:
        # Create LogManager with Redis + SQLite
        # Redis for realtime streaming, SQLite for persistence
        _log_manager = LogManager(redis_url=settings.REDIS_URL, db_path=settings.LOG_DB_PATH)

        # Register with logging handler
        LogManagerHandler.set_log_manager(_log_manager)

        # Register with admin router
        set_log_manager(_log_manager)

        # Start background log processor
        await _log_manager.start()

        logger.info(f"[Main] LogManager initialized with Redis={settings.REDIS_URL}, DB={settings.LOG_DB_PATH}")

    except Exception as e:
        logger.error(f"[Main] Failed to initialize LogManager: {e}")


async def shutdown_event() -> None:
    """Cleanup on app shutdown."""
    if _log_manager:
        await _log_manager.stop()
        logger.info("[Main] LogManager stopped")


app = FastAPI(title="Luxia Worker Service", version="1.0.0")

# Register startup/shutdown events
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# Include routers
app.include_router(pinecone_router, prefix="/worker", tags=["Pinecone"])
app.include_router(admin_router, tags=["Admin"])

logger.info("Luxia Worker Service initialized")
