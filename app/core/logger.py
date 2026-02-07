import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a structured logger for any module in the worker.

    Usage:
        logger = get_logger(__name__)

    Features:
    - Logs to stdout with formatted timestamp
    - Integrated with LogManager for realtime streaming and Neo4j persistence
    - Supports extra context (request_id, round_id, session_id)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] -> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

        # Add LogManagerHandler (will be initialized after app startup)
        try:
            from app.services.logging.log_handler import LogManagerHandler

            log_manager_handler = LogManagerHandler()
            logger.addHandler(log_manager_handler)
        except ImportError:
            # LogManager not available yet; will be added at startup
            pass

    return logger
