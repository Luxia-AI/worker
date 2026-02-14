import json
import logging
import os
import sys
from typing import Any, Dict


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
        logger.setLevel(logging.DEBUG if is_debug_enabled() else logging.INFO)

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


def is_debug_enabled() -> bool:
    return (os.getenv("DEBUG", "false") or "").strip().lower() in {"1", "true", "yes", "on"}


def truncate_list_payload(payload: Any, max_items: int | None = None) -> Any:
    if max_items is None:
        max_items = int(os.getenv("LOG_VALUE_SAMPLE_LIMIT", "5") or 5)
    hard_max = int(os.getenv("LOG_VALUE_MAX_ITEMS", "20") or 20)
    max_items = max(1, min(int(max_items), hard_max))
    if isinstance(payload, list):
        return [truncate_list_payload(x, max_items=max_items) for x in payload[:max_items]]
    if isinstance(payload, dict):
        return {k: truncate_list_payload(v, max_items=max_items) for k, v in payload.items()}
    return payload


def log_value_payload(
    logger: logging.Logger,
    phase: str,
    payload: Dict[str, Any],
    *,
    level: str = "info",
    debug_only: bool = False,
    sample_limit: int | None = None,
) -> None:
    if debug_only and not is_debug_enabled():
        return
    data = {"phase": phase, **(payload or {})}
    compact = truncate_list_payload(data, max_items=sample_limit)
    line = f"[PhaseOutput] {json.dumps(compact, ensure_ascii=True, default=str, separators=(',', ':'))}"
    if level == "debug":
        logger.debug(line)
    elif level == "warning":
        logger.warning(line)
    elif level == "error":
        logger.error(line)
    else:
        logger.info(line)
