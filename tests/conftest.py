"""
Pytest configuration and fixtures for test suite.

This module:
- Detects CI environment and skips tests requiring external services
- Provides common fixtures for async tests
- Sets up test logging
"""

import os
import socket

import pytest

# Detect CI environment
IS_CI = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") or os.environ.get("GITLAB_CI")


def is_redis_available():
    """Check if Redis is available on localhost:6379."""
    try:
        sock = socket.create_connection(("localhost", 6379), timeout=1)
        sock.close()
        return True
    except (socket.timeout, socket.error):
        return False


def is_neo4j_available():
    """Check if Neo4j is available on localhost:7687."""
    try:
        sock = socket.create_connection(("localhost", 7687), timeout=1)
        sock.close()
        return True
    except (socket.timeout, socket.error):
        return False


def is_pinecone_available():
    """Check if Pinecone API key is configured."""
    return bool(os.environ.get("PINECONE_API_KEY"))


def is_groq_available():
    """Check if Groq API key is configured."""
    return bool(os.environ.get("GROQ_API_KEY"))


def is_google_cse_available():
    """Check if Google CSE credentials are configured."""
    return bool(os.environ.get("GOOGLE_API_KEY")) and bool(os.environ.get("GOOGLE_CSE_ID"))


# Pytest markers for skipping
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "redis_required: mark test as requiring Redis connection")
    config.addinivalue_line("markers", "neo4j_required: mark test as requiring Neo4j connection")
    config.addinivalue_line("markers", "pinecone_required: mark test as requiring Pinecone API")
    config.addinivalue_line("markers", "groq_required: mark test as requiring Groq API")
    config.addinivalue_line("markers", "google_cse_required: mark test as requiring Google CSE API")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on environment."""
    for item in items:
        # Skip tests requiring Redis if not available
        if "redis_required" in item.keywords or "test_log_manager" in item.name:
            if not is_redis_available():
                item.add_marker(pytest.mark.skip(reason="Redis not available (expected in CI)"))

        # Skip tests requiring Neo4j if not available
        if "neo4j_required" in item.keywords:
            if not is_neo4j_available():
                item.add_marker(pytest.mark.skip(reason="Neo4j not available (expected in CI)"))

        # Skip tests requiring Pinecone if not configured
        if "pinecone_required" in item.keywords:
            if not is_pinecone_available():
                item.add_marker(pytest.mark.skip(reason="Pinecone API key not configured"))

        # Skip tests requiring Groq if not configured
        if "groq_required" in item.keywords:
            if not is_groq_available():
                item.add_marker(pytest.mark.skip(reason="Groq API key not configured"))

        # Skip tests requiring Google CSE if not configured
        if "google_cse_required" in item.keywords:
            if not is_google_cse_available():
                item.add_marker(pytest.mark.skip(reason="Google CSE credentials not configured"))

        # Skip E2E and integration tests in CI by default (unless explicitly enabled)
        if IS_CI:
            if "e2e" in item.keywords or "integration" in item.keywords:
                if not os.environ.get("RUN_INTEGRATION_TESTS"):
                    item.add_marker(pytest.mark.skip(reason="Integration tests skipped in CI by default"))


@pytest.fixture
def redis_available():
    """Fixture indicating if Redis is available."""
    return is_redis_available()


@pytest.fixture
def neo4j_available():
    """Fixture indicating if Neo4j is available."""
    return is_neo4j_available()


@pytest.fixture
def pinecone_available():
    """Fixture indicating if Pinecone is available."""
    return is_pinecone_available()


@pytest.fixture
def groq_available():
    """Fixture indicating if Groq is available."""
    return is_groq_available()


@pytest.fixture
def google_cse_available():
    """Fixture indicating if Google CSE is available."""
    return is_google_cse_available()
