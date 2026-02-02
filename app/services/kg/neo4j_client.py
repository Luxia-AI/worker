from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncGraphDatabase

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Timeout settings for Neo4j operations
NEO4J_CONNECTION_TIMEOUT = 10  # seconds to establish connection
NEO4J_QUERY_TIMEOUT = 15  # seconds per query execution


class Neo4jClient:
    """
    Async Neo4j driver wrapper for consistent sessions across the worker.

    - Provides async session creation with timeouts
    - Fails gracefully when Neo4j is unavailable
    - Logs failures
    - Used for both ingestion & retrieval
    """

    _driver = None
    _connection_failed = False  # Track if connection has failed to avoid repeated attempts

    def __init__(self) -> None:
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD

    async def _ensure_driver(self) -> Any:
        # If we've already failed to connect, don't retry immediately
        if self._connection_failed:
            raise ConnectionError("Neo4j connection previously failed, skipping")

        if self._driver is None:
            try:
                self._driver = AsyncGraphDatabase.driver(
                    str(self.uri),
                    auth=(str(self.user), str(self.password)),
                    max_connection_lifetime=300,  # 5 minutes
                    connection_timeout=NEO4J_CONNECTION_TIMEOUT,
                )
                # Test connection with timeout
                async with asyncio.timeout(NEO4J_CONNECTION_TIMEOUT):
                    async with self._driver.session() as session:
                        await session.run("RETURN 1 AS ok")
                logger.info("[Neo4jClient] Connected to Neo4j successfully")
                self._connection_failed = False
            except asyncio.TimeoutError:
                logger.error(f"[Neo4jClient] Connection timeout ({NEO4J_CONNECTION_TIMEOUT}s)")
                self._connection_failed = True
                self._driver = None
                raise ConnectionError("Neo4j connection timeout")
            except Exception as e:
                logger.error(f"[Neo4jClient] Failed to connect: {e}")
                self._connection_failed = True
                self._driver = None
                raise

        return self._driver

    @asynccontextmanager
    async def session(self) -> Any:
        """
        Async context manager returning a Neo4j session.
        Ensures driver initialized; handles session cleanup.
        Raises ConnectionError if Neo4j is unavailable.
        """
        driver = await self._ensure_driver()
        session = driver.session()
        try:
            yield session
        finally:
            await session.close()

    async def execute(
        self, query: str, params: dict | None = None, timeout: float = NEO4J_QUERY_TIMEOUT
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results as list of dicts.

        Args:
            query: Cypher query string
            params: Query parameters dict
            timeout: Query timeout in seconds (default 15s)

        Returns:
            List of result records as dictionaries

        Raises:
            asyncio.TimeoutError: If query exceeds timeout
            ConnectionError: If Neo4j is unavailable
        """
        try:
            async with asyncio.timeout(timeout):
                async with self.session() as session:
                    result = await session.run(query, params or {})
                    records = await result.data()
                    return records
        except asyncio.TimeoutError:
            logger.error(f"[Neo4jClient] Query timeout ({timeout}s)")
            raise

    async def close(self) -> None:
        """
        Cleanly close the global Neo4j driver.
        """
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._connection_failed = False
            logger.info("[Neo4jClient] Driver closed")

    @classmethod
    def reset_connection_state(cls) -> None:
        """Reset connection failure state to allow retry."""
        cls._connection_failed = False
