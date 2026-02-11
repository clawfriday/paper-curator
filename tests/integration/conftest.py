"""Fixtures for integration tests.

Key behaviour:
  1. Before any integration test runs, switch the backend to a fresh test database
     (paper_curator_test) via the /db/init and /db/switch API.
  2. After all integration tests finish, switch back to the production database.
  3. Each test run starts with a clean, empty test database.
"""
import os
import uuid

import pytest
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")
TEST_DB_NAME = os.environ.get("TEST_DB_NAME", "paper_curator_test")
TEST_STORAGE = os.path.join(os.path.dirname(__file__), "..", "storage")

# ---------------------------------------------------------------------------
# Session-scoped: database switching (runs once per pytest session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def switch_to_test_database():
    """Create (if needed) and switch the backend to the test database.

    Yields control to the test session, then switches back to production.
    """
    # 0. Remember the current (production) database
    try:
        status = requests.get(f"{BACKEND_URL}/db/status", timeout=10).json()
        prod_db = status.get("database", "paper_curator")
    except Exception:
        pytest.skip("Backend not available — cannot switch database")
        return

    # 1. Create / reset test database (drop + recreate for a clean slate)
    print(f"\n{'='*60}")
    print(f"  SETUP: Creating test DB '{TEST_DB_NAME}' (drop_existing=true)")
    print(f"{'='*60}")
    init_resp = requests.post(
        f"{BACKEND_URL}/db/init",
        json={"database": TEST_DB_NAME, "drop_existing": True},
        timeout=30,
    )
    assert init_resp.status_code == 200, f"Failed to init test DB: {init_resp.text}"
    print(f"  -> {init_resp.json()}")

    # 2. Switch backend to test database
    switch_resp = requests.post(
        f"{BACKEND_URL}/db/switch",
        json={"database": TEST_DB_NAME},
        timeout=10,
    )
    assert switch_resp.status_code == 200, f"Failed to switch to test DB: {switch_resp.text}"
    switch_data = switch_resp.json()
    print(f"  -> Switched: {switch_data['previous_database']} → {switch_data['current_database']}")
    print(f"  -> Paper count in test DB: {switch_data['paper_count']}")
    print(f"{'='*60}\n")

    # --- RUN ALL INTEGRATION TESTS ---
    yield

    # 3. Switch back to production database
    print(f"\n{'='*60}")
    print(f"  TEARDOWN: Switching back to production DB '{prod_db}'")
    print(f"{'='*60}")
    try:
        restore_resp = requests.post(
            f"{BACKEND_URL}/db/switch",
            json={"database": prod_db},
            timeout=10,
        )
        if restore_resp.status_code == 200:
            print(f"  -> Restored to {restore_resp.json()['current_database']}")
        else:
            print(f"  -> WARNING: Failed to restore: {restore_resp.text}")
    except Exception as e:
        print(f"  -> WARNING: Failed to restore production DB: {e}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend_url():
    """Backend URL."""
    return BACKEND_URL


@pytest.fixture(scope="module")
def backend_available(backend_url):
    """Ensure backend is running and connected to the test database."""
    try:
        resp = requests.get(f"{backend_url}/health", timeout=10)
        if resp.status_code != 200:
            pytest.skip("Backend not healthy")
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not available")

    # Verify we're on the test database
    status = requests.get(f"{backend_url}/db/status", timeout=10).json()
    assert status.get("database") == TEST_DB_NAME, (
        f"Backend is on '{status.get('database')}', expected '{TEST_DB_NAME}'. "
        "Did the database switch fail?"
    )
    return True


# ---------------------------------------------------------------------------
# Function-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_storage_dir():
    """Test storage directory."""
    os.makedirs(f"{TEST_STORAGE}/downloads/local", exist_ok=True)
    return TEST_STORAGE


@pytest.fixture
def sample_pdf_path(test_storage_dir):
    """Path to sample PDF if available."""
    from pathlib import Path
    pdf_path = Path(test_storage_dir) / "downloads" / "local" / "1706.03762.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF not available")
    return str(pdf_path)


@pytest.fixture
def unique_test_id():
    """Generate unique ID for test isolation."""
    return str(uuid.uuid4())[:8]
