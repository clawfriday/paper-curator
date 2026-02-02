import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add src/backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

from app import app


@pytest.fixture
def client():
    """Create a test client that returns HTTP responses instead of raising exceptions."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_arxiv_id():
    """Sample arXiv ID for testing (Attention Is All You Need paper)."""
    return "1706.03762"
