"""Test classification endpoint."""
import os

import pytest


@pytest.mark.external
def test_classify_endpoint(client):
    """Run POST /papers/classify and verify response structure."""
    response = client.post("/papers/classify")
    if response.status_code in [502, 503]:
        if os.environ.get("REQUIRE_EXTERNAL_ENDPOINTS"):
            pytest.fail("LLM/Embedding endpoint not available (REQUIRE_EXTERNAL_ENDPOINTS=1)")
        pytest.skip("LLM/Embedding endpoint not available")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "papers_classified" in data
    assert "clusters_created" in data
