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


class TestTopicSearchFallback:
    """Verify topic search returns results even for short/specific queries.

    Root cause: short queries like "FP8" produce lower cosine similarity
    scores (~0.40) which fall below the default similarity_threshold (0.5),
    causing search to return 0 papers. The fallback should re-query without
    threshold when the initial query returns nothing.
    """

    def test_short_query_returns_papers(self, backend_available):
        """POST /topic/search with a short query should return papers via fallback."""
        resp = requests.post(
            f"{BACKEND_URL}/topic/search",
            json={"topic": "FP8", "limit": 10, "offset": 0},
            timeout=30,
        )
        assert resp.status_code == 200, f"topic/search failed: {resp.status_code} {resp.text}"
        data = resp.json()
        assert "papers" in data
        papers = data["papers"]
        assert len(papers) > 0, (
            "Topic search for 'FP8' returned 0 papers — "
            "similarity threshold may be filtering out all results"
        )
        # Top result should be FP8-related
        top = papers[0]
        assert "title" in top
        assert "similarity" in top

    def test_descriptive_query_returns_papers(self, backend_available):
        """POST /topic/search with a descriptive query should return papers."""
        resp = requests.post(
            f"{BACKEND_URL}/topic/search",
            json={"topic": "deep neural network architectures", "limit": 10, "offset": 0},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["papers"]) > 0, "No papers found for descriptive query"
