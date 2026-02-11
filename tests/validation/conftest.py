"""Fixtures for validation tests."""
import os
import pytest
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")


@pytest.fixture(scope="module")
def backend_url():
    """Backend URL."""
    return BACKEND_URL


@pytest.fixture(scope="module")
def backend_available(backend_url):
    """Check if backend is available."""
    try:
        resp = requests.get(f"{backend_url}/health", timeout=5)
        if resp.status_code != 200:
            pytest.skip("Backend not healthy")
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not available")
    return True
