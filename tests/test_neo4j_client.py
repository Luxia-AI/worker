from unittest.mock import AsyncMock, patch

import pytest

from app.services.kg.neo4j_client import Neo4jClient


@pytest.mark.asyncio
async def test_neo4j_client_connect_and_session():
    client = Neo4jClient()

    with patch("neo4j.AsyncGraphDatabase.driver") as mock_driver:
        mock_session = AsyncMock()
        mock_driver.return_value.session.return_value = mock_session
        mock_session.run = AsyncMock()

        # test session creation
        async with client.session() as s:
            assert s is mock_session

        # ensure driver initialized only once
        await client._ensure_driver()
        mock_driver.assert_called_once()
