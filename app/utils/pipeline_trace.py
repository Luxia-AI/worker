import json
import os
import re
import time
from typing import Any

DEBUG = os.getenv("DEBUG", "false").strip().lower() == "true"

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|token|secret|password|authorization|auth|credential|connectionstring)",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = [
    re.compile(r"\b(gsk_[A-Za-z0-9_\-]{12,})\b"),
    re.compile(r"\b(sk-[A-Za-z0-9]{12,})\b"),
    re.compile(r"\b(AIza[0-9A-Za-z_\-]{12,})\b"),
    re.compile(r"\b(pcsk_[A-Za-z0-9_\-]{12,})\b"),
    re.compile(r"\b(rediss?://[^\s]+)\b", re.IGNORECASE),
]


def _mask_string(value: str) -> str:
    text = str(value)
    masked = text
    for pattern in _SECRET_VALUE_PATTERNS:
        masked = pattern.sub("***MASKED***", masked)
    return masked


def _sanitize(data: Any, parent_key: str = "") -> Any:
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            key_str = str(key)
            if _SECRET_KEY_RE.search(key_str):
                out[key_str] = "***MASKED***"
            else:
                out[key_str] = _sanitize(value, parent_key=key_str)
        return out
    if isinstance(data, list):
        return [_sanitize(item, parent_key=parent_key) for item in data]
    if isinstance(data, tuple):
        return tuple(_sanitize(item, parent_key=parent_key) for item in data)
    if isinstance(data, str):
        if _SECRET_KEY_RE.search(parent_key):
            return "***MASKED***"
        return _mask_string(data)
    return data


def trace(stage: str, data: Any = None) -> None:
    """
    Print structured pipeline trace output when DEBUG mode is enabled.
    """
    if not DEBUG:
        return

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 80)
    print(f"[PIPELINE TRACE] {ts}")
    print(f"Stage: {stage}")

    if data is not None:
        safe = _sanitize(data)
        try:
            print(json.dumps(safe, indent=2, default=str))
        except Exception:
            print(safe)

    print("=" * 80)
