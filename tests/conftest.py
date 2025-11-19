"""
Pytest configuration and fixtures.

This module configures pytest to:
- Use lightweight embedding models for tests (all-MiniLM-L6-v2)
- Mock external APIs (GROQ, Google Search) for testing in CI environments
- Mark slow integration tests for selective running
- Provide environment setup for testing
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test (requires external services like Pinecone)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )


def _is_ci_environment() -> bool:
    """
    Detect if running in a CI environment.
    Checks for common CI environment variables.
    """
    ci_indicators = [
        "CI",  # GitHub Actions, GitLab CI, CircleCI
        "CONTINUOUS_INTEGRATION",  # Travis CI
        "BUILD_ID",  # Jenkins
        "RUN_ID",  # GitHub Actions
    ]
    return any(os.getenv(indicator) for indicator in ci_indicators)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Set up test environment variables for testing.

    In CI environments:
    - Sets dummy API keys to allow services to initialize
    - Enables lightweight embedding model detection

    In local environments:
    - Uses real API keys from environment if available
    """
    if _is_ci_environment():
        # Set dummy API keys only in CI so services can initialize
        if not os.getenv("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = "test-groq-key"
        if not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = "test-google-key"
        if not os.getenv("GOOGLE_CSE_ID"):
            os.environ["GOOGLE_CSE_ID"] = "test-cse-id"


@pytest.fixture(autouse=True)
def mock_groq_client():
    """
    Mock AsyncGroq API client in CI environments to avoid actual API calls.
    Skipped in local development environments.
    """
    if not _is_ci_environment():
        yield
        return

    # Create mock response
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"entities": ["vitamin d", "cancer"]})

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # Mock AsyncGroq
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("groq.AsyncGroq") as mock_groq_class:
        mock_groq_class.return_value = mock_client
        yield mock_client


@pytest.fixture(autouse=True)
def mock_google_client():
    """
    Mock aiohttp ClientSession in CI environments to avoid actual Google API calls.
    Skipped in local development environments.
    """
    if not _is_ci_environment():
        yield
        return

    # Create mock response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "items": [
                {"link": "https://www.who.int/health-topics/cancer"},
                {"link": "https://www.cdc.gov/cancer/prevention"},
                {"link": "https://www.nih.gov/news-events/cancer-research"},
            ]
        }
    )

    # Create mock session
    mock_session = MagicMock()
    mock_session.get = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession") as mock_client_session:
        mock_client_session.return_value = mock_session
        yield mock_session
