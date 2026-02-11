"""Fixtures for integration tests."""
import os
import pytest
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")
TEST_STORAGE = os.path.join(os.path.dirname(__file__), "..", "storage")
TEST_DB_PORT = os.environ.get("PGPORT", "5433")


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
def test_storage_dir():
    """Test storage directory."""
    os.makedirs(f"{TEST_STORAGE}/downloads", exist_ok=True)
    return TEST_STORAGE


@pytest.fixture
def sample_pdf_path(test_storage_dir):
    """Path to sample PDF if available."""
    from pathlib import Path
    pdf_path = Path(test_storage_dir) / "downloads" / "1706.03762.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF not available - run make test-download-samples")
    return str(pdf_path)


@pytest.fixture
def unique_test_id():
    """Generate unique ID for test isolation."""
    import uuid
    return str(uuid.uuid4())[:8]
