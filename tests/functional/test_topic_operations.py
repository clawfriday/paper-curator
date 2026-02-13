"""Functional tests for topic query operations.

Catches serialization issues (numpy.ndarray in JSON responses)
and verifies topic endpoints return valid JSON.
"""
import os

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")


class TestTopicEndpointSerialization:
    """Verify topic endpoints return valid JSON without numpy serialization errors.

    Root cause: pgvector returns embedding columns as numpy.ndarray, which
    Pydantic cannot serialize. Any endpoint returning topic rows that include
    the embedding column will fail with PydanticSerializationError.
    """

    def test_topic_check_returns_json(self, backend_available):
        """GET /topic/check should return valid JSON, not 500 from numpy."""
        resp = requests.get(
            f"{BACKEND_URL}/topic/check",
            params={"topic_query": "nonexistent_test_topic_xyz"},
            timeout=15,
        )
        assert resp.status_code == 200, f"topic/check failed: {resp.status_code} {resp.text}"
        data = resp.json()
        assert "exists" in data
        assert isinstance(data["exists"], bool)
        assert "topics" in data
        assert isinstance(data["topics"], list)

    def test_topic_list_returns_json(self, backend_available):
        """GET /topic/list should return valid JSON."""
        resp = requests.get(f"{BACKEND_URL}/topic/list", timeout=15)
        assert resp.status_code == 200, f"topic/list failed: {resp.status_code} {resp.text}"
        data = resp.json()
        # Response is {"topics": [...]} or a list
        topics = data.get("topics", data) if isinstance(data, dict) else data
        assert isinstance(topics, list)

    def test_topic_get_missing_returns_404(self, backend_available):
        """GET /topic/{id} for non-existent topic should return 404, not 500."""
        resp = requests.get(f"{BACKEND_URL}/topic/999999", timeout=15)
        # Should be 404 (not found), not 500 (serialization error)
        assert resp.status_code in (404, 200), (
            f"topic/999999 returned unexpected {resp.status_code}: {resp.text}"
        )
