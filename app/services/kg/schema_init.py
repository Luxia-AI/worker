from __future__ import annotations

from app.core.logger import get_logger
from app.services.kg.neo4j_client import Neo4jClient

logger = get_logger(__name__)


class KGSchemaInitializer:
    def __init__(self):
        self.client = Neo4jClient()

    async def initialize(self):
        constraints = [
            """
            CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE e.name IS UNIQUE
            """,
            """
            CREATE INDEX rel_relation_idx IF NOT EXISTS
            FOR ()-[r:RELATION]-()
            ON (r.relation)
            """,
            """
            CREATE INDEX entity_name_idx IF NOT EXISTS
            FOR (e:Entity)
            ON (e.name)
            """,
        ]

        async with self.client.session() as session:
            for c in constraints:
                try:
                    await session.run(c)
                    logger.info(f"[KGSchemaInitializer] Applied: {c[:40]}...")
                except Exception as e:
                    logger.error(f"[KGSchemaInitializer] Failed: {e}")
