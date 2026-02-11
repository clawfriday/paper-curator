"""Shared pytest fixtures for all tests."""
import sys
from pathlib import Path

import pytest

# Add src/backend to path for imports (but don't import app yet)
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))


@pytest.fixture
def client():
    """Create a test client that returns HTTP responses instead of raising exceptions.
    
    Note: This fixture lazily imports the app to avoid import errors in tests
    that don't need the TestClient (e.g., connectivity tests using requests).
    """
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_arxiv_id():
    """Sample arXiv ID for testing (Attention Is All You Need paper)."""
    return "1706.03762"
