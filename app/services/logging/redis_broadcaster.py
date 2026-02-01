"""
Redis-based realtime log streaming for admin dashboard.
Uses Redis pub/sub for efficient message broadcasting to multiple subscribers.
"""

import json
from typing import Any, Callable, Dict, Optional

import redis.asyncio as redis

from app.core.logger import get_logger

logger = get_logger(__name__)


class RedisLogBroadcaster:
    """
    Publish/subscribe logs to Redis for realtime streaming to WebSocket clients.
    One Redis channel per request_id for efficient filtering.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.channel_prefix = "logs:"  # e.g., "logs:claim-123"

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            # Azure Redis requires SSL (rediss:// or port 6380)
            ssl_enabled = self.redis_url.startswith("rediss://") or ":6380" in self.redis_url

            if ssl_enabled:
                # Azure Redis with SSL - use ssl_cert_reqs for rediss:// URLs
                self.redis_client = await redis.from_url(
                    self.redis_url,
                    encoding="utf8",
                    decode_responses=True,
                    ssl_cert_reqs=None,  # Disable cert verification for Azure
                    socket_timeout=30.0,
                    socket_connect_timeout=30.0,
                )
            else:
                self.redis_client = await redis.from_url(
                    self.redis_url,
                    encoding="utf8",
                    decode_responses=True,
                )

            ping_result = self.redis_client.ping()  # type: ignore
            if hasattr(ping_result, "__await__"):
                await ping_result
            logger.info(f"[RedisLogBroadcaster] Connected to Redis (ssl={ssl_enabled})")
        except Exception as e:
            logger.error(f"[RedisLogBroadcaster] Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("[RedisLogBroadcaster] Disconnected from Redis")

    async def publish(self, log_record: Dict[str, Any]) -> None:
        """
        Publish log to Redis channels.
        Publishes to:
        - logs:all - all logs
        - logs:{request_id} - logs for specific request (if request_id present)
        - logs:{level} - logs for specific level

        Args:
            log_record: Log record dict with id, level, message, module, etc.
        """
        if not self.redis_client:
            return

        try:
            message = json.dumps(log_record)

            # Always publish to "all" channel
            await self.redis_client.publish("logs:all", message)

            # Publish to request-specific channel if available
            if log_record.get("request_id"):
                channel = f"{self.channel_prefix}{log_record['request_id']}"
                await self.redis_client.publish(channel, message)

            # Publish to level-specific channel
            channel = f"logs:{log_record['level']}"
            await self.redis_client.publish(channel, message)

        except Exception as e:
            logger.error(f"[RedisLogBroadcaster] Failed to publish log: {e}")

    async def subscribe(
        self,
        channel: str,
        callback: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """
        Subscribe to a Redis channel and call callback for each message.

        Args:
            channel: Channel name (e.g., "logs:all", "logs:claim-123", "logs:ERROR")
            callback: Async function to call with parsed log record

        Usage:
            async def on_log(log_record):
                print(f"Got log: {log_record['message']}")

            await broadcaster.subscribe("logs:all", on_log)
        """
        if not self.redis_client:
            logger.error("[RedisLogBroadcaster] Redis not connected")
            return

        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(channel)
            logger.info(f"[RedisLogBroadcaster] Subscribed to channel: {channel}")

            # Listen for messages
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        log_record = json.loads(message["data"])
                        await callback(log_record)
                    except Exception as e:
                        logger.error(f"[RedisLogBroadcaster] Error processing message: {e}")

        except Exception as e:
            logger.error(f"[RedisLogBroadcaster] Subscription error: {e}")
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    async def get_active_channels(self) -> int:
        """Get count of active subscriptions (for monitoring)."""
        if not self.redis_client:
            return 0

        try:
            channels = await self.redis_client.pubsub_channels()
            return len(channels)
        except Exception:
            return 0

    async def delete_channel(self, channel: str) -> None:
        """Delete a channel (cleanup)."""
        if not self.redis_client:
            return

        try:
            await self.redis_client.delete(channel)
        except Exception as e:
            logger.error(f"[RedisLogBroadcaster] Failed to delete channel: {e}")
