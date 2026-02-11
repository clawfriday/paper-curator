"""Fixtures for functional tests."""
import os
import pytest
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")
TEST_STORAGE = os.path.join(os.path.dirname(__file__), "..", "storage")


@pytest.fixture(scope="module")
def backend_url():
    """Backend URL."""
    return BACKEND_URL


@pytest.fixture(scope="module")
def backend_available(backend_url):
    """Ensure backend is running."""
    try:
        resp = requests.get(f"{backend_url}/health", timeout=10)
        if resp.status_code != 200:
            pytest.skip("Backend not healthy")
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not available")
    return True


@pytest.fixture
def sample_arxiv_id():
    """Attention Is All You Need paper."""
    return "1706.03762"


@pytest.fixture
def sample_paper_title():
    """Title for the sample paper."""
    return "Attention Is All You Need"


@pytest.fixture
def test_storage_dir():
    """Test storage directory."""
    os.makedirs(f"{TEST_STORAGE}/downloads", exist_ok=True)
    return TEST_STORAGE
