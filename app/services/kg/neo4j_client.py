from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncGraphDatabase

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """
    Async Neo4j driver wrapper for consistent sessions across the worker.

    - Provides async session creation
    - Retries connections
    - Logs failures
    - Used for both ingestion & retrieval
    """

    _driver = None

    def __init__(self) -> None:
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD

    async def _ensure_driver(self) -> Any:
        if self._driver is None:
            try:
                self._driver = AsyncGraphDatabase.driver(
                    str(self.uri),
                    auth=(str(self.user), str(self.password)),
                    max_connection_lifetime=3600,
                )
                # Test connection
                async with self._driver.session() as session:
                    await session.run("RETURN 1 AS ok")
                logger.info("[Neo4jClient] Connected to Neo4j successfully")
            except Exception as e:
                logger.error(f"[Neo4jClient] Failed to connect: {e}")
                raise

        return self._driver

    @asynccontextmanager
    async def session(self) -> Any:
        """
        Async context manager returning a Neo4j session.
        Ensures driver initialized; handles session cleanup.
        """
        driver = await self._ensure_driver()
        session = driver.session()
        try:
            yield session
        finally:
            await session.close()

    async def execute(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results as list of dicts.

        Args:
            query: Cypher query string
            params: Query parameters dict

        Returns:
            List of result records as dictionaries
        """
        async with self.session() as session:
            result = await session.run(query, params or {})
            records = await result.data()
            return records

    async def close(self) -> None:
        """
        Cleanly close the global Neo4j driver.
        """
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("[Neo4jClient] Driver closed")
