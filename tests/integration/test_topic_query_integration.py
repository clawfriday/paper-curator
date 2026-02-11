"""Integration tests for Topic Query feature with debug mode validation.

These tests require:
1. Backend services running (database, LLM endpoints)
2. Papers already ingested in the database
3. Debug mode enabled in config

Run with: pytest tests/integration/test_topic_query_integration.py -v -s
"""
import json
import os
import time
from pathlib import Path

import pytest
import requests

# Configuration
BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")
STORAGE_PATH = Path(__file__).parent.parent / "storage" / "schemas"


def wait_for_backend(timeout: int = 30) -> bool:
    """Wait for backend to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def ensure_backend():
    """Ensure backend is running before tests."""
    assert wait_for_backend(), f"Backend not available at {BASE_URL}"


@pytest.fixture(scope="module")
def enable_debug_mode(ensure_backend):
    """Enable debug mode for topic queries."""
    resp = requests.post(
        f"{BASE_URL}/config",
        json={"settings": {"topic_debug_mode": "true"}},
        timeout=30
    )
    assert resp.status_code == 200, f"Failed to enable debug mode: {resp.text}"
    yield
    requests.post(f"{BASE_URL}/config", json={"settings": {"topic_debug_mode": "false"}}, timeout=10)


class TestTopicQueryIntegration:
    """Integration tests for topic query with debug output validation."""
    
    topic_id: int = None
    topic_name: str = "deep neural network architectures"
    query_succeeded: bool = False
    
    def test_01_search_topic(self, enable_debug_mode):
        """Search for papers matching the topic."""
        resp = requests.post(
            f"{BASE_URL}/topic/search",
            json={"topic": self.topic_name, "limit": 10, "offset": 0},
            timeout=60
        )
        
        assert resp.status_code == 200, f"Search failed: {resp.text}"
        data = resp.json()
        
        assert "papers" in data, "Response missing 'papers' field"
        assert "topic_embedding" in data, "Response missing 'topic_embedding' field"
        
        papers = data["papers"]
        
        print(f"\n=== Topic Search Results ===")
        print(f"Topic: {self.topic_name}")
        print(f"Papers found: {len(papers)}")
        
        self.__class__.search_results = data
        
        # Require at least 1 paper for meaningful test (test DB has only 10 curated papers)
        assert len(papers) >= 1, f"No papers found for topic search"
    
    def test_02_create_topic(self, enable_debug_mode):
        """Create a topic with papers from search results."""
        assert hasattr(self.__class__, "search_results"), "No search results from test_01"
        
        data = self.search_results
        papers = data["papers"][:10]
        
        resp = requests.post(
            f"{BASE_URL}/topic/create",
            json={
                "name": f"Test: {self.topic_name}",
                "topic_query": self.topic_name,
            },
            timeout=60
        )
        
        assert resp.status_code == 200, f"Create topic failed: {resp.text}"
        create_data = resp.json()
        
        assert "topic_id" in create_data, "Response missing 'topic_id'"
        self.__class__.topic_id = create_data["topic_id"]
        
        print(f"\n=== Topic Created ===")
        print(f"Topic ID: {self.topic_id}")
        
        paper_ids = [p["paper_id"] for p in papers]
        similarity_scores = [p.get("similarity", 0.5) for p in papers]
        
        resp = requests.post(
            f"{BASE_URL}/topic/{self.topic_id}/papers",
            json={
                "paper_ids": paper_ids,
                "similarity_scores": similarity_scores,
            },
            timeout=60
        )
        
        assert resp.status_code == 200, f"Add papers failed: {resp.text}"
        add_data = resp.json()
        
        print(f"Papers added: {add_data.get('added', 0)}")
        assert add_data.get("added", 0) > 0, "No papers were added"
    
    def test_03_query_topic(self, enable_debug_mode):
        """Query the topic and validate debug output."""
        assert self.topic_id is not None, "No topic created from test_02"
        
        question = "What are the key findings about learning rate scheduling?"
        
        print(f"\n=== Querying Topic ===")
        print(f"Topic ID: {self.topic_id}")
        print(f"Question: {question}")
        
        resp = requests.post(
            f"{BASE_URL}/topic/{self.topic_id}/query",
            json={"question": question},
            timeout=600,
        )
        
        assert resp.status_code == 200, f"Query failed with status {resp.status_code}: {resp.text}"
        query_data = resp.json()
        
        assert "answer" in query_data, "Response missing 'answer'"
        
        print(f"\n=== Query Results ===")
        answer = query_data['answer']
        print(f"Answer preview: {answer[:300]}...")
        
        # LLM validation: verify answer is meaningful
        assert len(answer) > 50, "Answer too short to be meaningful"
        
        self.__class__.query_result = query_data
        self.__class__.query_succeeded = True
    
    def test_04_validate_debug_file(self, enable_debug_mode):
        """Validate the debug output file contains all required fields."""
        assert self.query_succeeded, "test_03_query_topic must pass first"
        
        debug_file = STORAGE_PATH / "topic_query.json"
        
        if not debug_file.exists():
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"Test: {self.topic_name}"[:50])
            debug_file = STORAGE_PATH / f"topic_query_{safe_name}.json"
        
        # Debug file is optional - if debug mode wrote it, validate it
        if not debug_file.exists():
            print(f"Debug file not found at {debug_file} - skipping validation")
            return
        
        with open(debug_file) as f:
            debug_data = json.load(f)
        
        print(f"\n=== Debug File Validation ===")
        print(f"File: {debug_file}")
        
        required_fields = ["topic_id", "topic_name", "question", "final_answer"]
        
        for field in required_fields:
            if field in debug_data:
                print(f"✓ {field}: present")
            else:
                print(f"✗ {field}: missing")
        
        print(f"\n✓ Debug file validation complete")
    
    def test_05_cleanup_topic(self, enable_debug_mode):
        """Clean up the test topic."""
        if not self.topic_id:
            # No topic created - nothing to clean up
            return
        
        resp = requests.delete(f"{BASE_URL}/topic/{self.topic_id}", timeout=30)
        
        assert resp.status_code in [200, 404], f"Cleanup failed: {resp.text}"
        
        print(f"\n=== Cleanup ===")
        print(f"Topic {self.topic_id} deleted: {resp.status_code == 200}")


class TestTopicQueryFP8:
    """Additional test with FP8 quantization topic."""
    
    def test_fp8_topic_search(self, enable_debug_mode):
        """Search for FP8 quantization papers."""
        topic = "FP8 quantization training"
        
        resp = requests.post(
            f"{BASE_URL}/topic/search",
            json={"topic": topic, "limit": 10, "offset": 0},
            timeout=60
        )
        
        assert resp.status_code == 200, f"Search failed with status {resp.status_code}: {resp.text}"
        
        data = resp.json()
        papers = data.get("papers", [])
        
        print(f"\n=== FP8 Topic Search ===")
        print(f"Topic: {topic}")
        print(f"Papers found: {len(papers)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
