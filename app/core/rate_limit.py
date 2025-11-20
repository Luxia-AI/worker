import asyncio
import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class AsyncRateLimiter:
    """
    Async Token Bucket Rate Limiter.
    Ensures that no more than `max_calls` occur within `period` seconds.

    Example:
        limiter = AsyncRateLimiter(max_calls=5, period=1)
        await limiter.acquire()  # blocks until allowed
    """

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self._tokens: float = float(max_calls)
        self._lock = asyncio.Lock()
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            refill_amount = (elapsed / self.period) * self.max_calls
            self._tokens = min(self.max_calls, self._tokens + refill_amount)
            self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                # Not enough tokens — wait a little
                await asyncio.sleep(0.01)


# Global registry for decorators
_rate_limiters: Dict[str, AsyncRateLimiter] = {}


def throttled(
    limit: int,
    period: float,
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator for async throttling.

    Args:
        limit: max calls allowed within `period` seconds
        period: sliding window duration in seconds
        name: optional bucket key (function name by default)

    Usage:
        @throttled(limit=5, period=1)
        async def google_search(...):
            ...
    """

    def decorator(func: F) -> F:
        limiter_name = name or func.__name__
        if limiter_name not in _rate_limiters:
            _rate_limiters[limiter_name] = AsyncRateLimiter(limit, period)

        limiter = _rate_limiters[limiter_name]

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
