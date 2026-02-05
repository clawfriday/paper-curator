"""Integration tests for Topic Query feature with debug mode validation.

These tests require:
1. Backend services running (database, LLM endpoints)
2. Papers already ingested in the database
3. Debug mode enabled in config

Run with: pytest tests/test_topic_query_integration.py -v -s
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
    if not wait_for_backend():
        pytest.skip("Backend not available")


@pytest.fixture(scope="module")
def enable_debug_mode(ensure_backend):
    """Enable debug mode for topic queries."""
    # Set debug mode via settings API
    resp = requests.post(
        f"{BASE_URL}/config",
        json={"settings": {"topic_debug_mode": "true"}}
    )
    assert resp.status_code == 200, f"Failed to enable debug mode: {resp.text}"
    yield
    # Optionally disable after tests
    requests.post(f"{BASE_URL}/config", json={"settings": {"topic_debug_mode": "false"}})


class TestTopicQueryIntegration:
    """Integration tests for topic query with debug output validation."""
    
    topic_id: int = None
    topic_name: str = "learning dynamics"
    
    def test_01_search_topic(self, enable_debug_mode):
        """Search for papers matching the topic."""
        resp = requests.post(
            f"{BASE_URL}/topic/search",
            json={"topic": self.topic_name, "limit": 10, "offset": 0}
        )
        
        assert resp.status_code == 200, f"Search failed: {resp.text}"
        data = resp.json()
        
        # Validate response structure
        assert "papers" in data, "Response missing 'papers' field"
        assert "topic_embedding" in data, "Response missing 'topic_embedding' field"
        
        papers = data["papers"]
        topic_embedding = data["topic_embedding"]
        
        print(f"\n=== Topic Search Results ===")
        print(f"Topic: {self.topic_name}")
        print(f"Papers found: {len(papers)}")
        print(f"Topic embedding dimension: {len(topic_embedding)}")
        
        # Validate embedding
        assert len(topic_embedding) > 0, "Topic embedding is empty"
        assert all(isinstance(x, (int, float)) for x in topic_embedding[:10]), "Invalid embedding values"
        
        # Validate papers have required fields
        if papers:
            for i, paper in enumerate(papers[:3]):
                print(f"\nPaper {i+1}:")
                print(f"  ID: {paper.get('paper_id')}")
                print(f"  arXiv: {paper.get('arxiv_id')}")
                print(f"  Title: {paper.get('title', 'N/A')[:60]}...")
                print(f"  Similarity: {paper.get('similarity', 'N/A')}")
                
                assert "paper_id" in paper, "Paper missing 'paper_id'"
                assert "similarity" in paper, "Paper missing 'similarity'"
        
        # Store for next tests
        self.__class__.search_results = data
        
        if len(papers) < 3:
            pytest.skip(f"Not enough papers found ({len(papers)}), need at least 3")
    
    def test_02_create_topic(self, enable_debug_mode):
        """Create a topic with papers from search results."""
        if not hasattr(self.__class__, "search_results"):
            pytest.skip("No search results from previous test")
        
        data = self.search_results
        papers = data["papers"][:10]  # Take top 10
        
        # Create topic - API generates embedding from topic_query
        resp = requests.post(
            f"{BASE_URL}/topic/create",
            json={
                "name": f"Test: {self.topic_name}",
                "topic_query": self.topic_name,
            }
        )
        
        assert resp.status_code == 200, f"Create topic failed: {resp.text}"
        create_data = resp.json()
        
        assert "topic_id" in create_data, "Response missing 'topic_id'"
        self.__class__.topic_id = create_data["topic_id"]
        
        print(f"\n=== Topic Created ===")
        print(f"Topic ID: {self.topic_id}")
        
        # Add papers to topic
        paper_ids = [p["paper_id"] for p in papers]
        similarity_scores = [p.get("similarity", 0.5) for p in papers]
        
        resp = requests.post(
            f"{BASE_URL}/topic/{self.topic_id}/papers",
            json={
                "paper_ids": paper_ids,
                "similarity_scores": similarity_scores,
            }
        )
        
        assert resp.status_code == 200, f"Add papers failed: {resp.text}"
        add_data = resp.json()
        
        print(f"Papers added: {add_data.get('added', 0)}")
        assert add_data.get("added", 0) > 0, "No papers were added"
        
        self.__class__.papers_added = len(paper_ids)
    
    def test_03_query_topic(self, enable_debug_mode):
        """Query the topic and validate debug output."""
        if not self.topic_id:
            pytest.skip("No topic created from previous test")
        
        question = "What are the key findings about learning rate scheduling and warmup strategies?"
        
        print(f"\n=== Querying Topic ===")
        print(f"Topic ID: {self.topic_id}")
        print(f"Question: {question}")
        
        resp = requests.post(
            f"{BASE_URL}/topic/{self.topic_id}/query",
            json={"question": question},
            timeout=300,  # 5 minutes timeout for RAG queries
        )
        
        assert resp.status_code == 200, f"Query failed: {resp.text}"
        query_data = resp.json()
        
        # Validate response
        assert "answer" in query_data, "Response missing 'answer'"
        assert "papers_queried" in query_data, "Response missing 'papers_queried'"
        assert "successful_queries" in query_data, "Response missing 'successful_queries'"
        
        print(f"\n=== Query Results ===")
        print(f"Papers queried: {query_data['papers_queried']}")
        print(f"Successful queries: {query_data['successful_queries']}")
        print(f"Answer preview: {query_data['answer'][:300]}...")
        
        # Validate paper_responses in debug mode
        if "paper_responses" in query_data:
            print(f"\n=== Paper Responses (Debug) ===")
            for i, pr in enumerate(query_data["paper_responses"][:3]):
                print(f"\nPaper {i+1}: {pr.get('title', 'N/A')[:50]}...")
                print(f"  Success: {pr.get('success')}")
                print(f"  Chunks retrieved: {pr.get('chunks_retrieved', 'N/A')}")
                if pr.get("response"):
                    print(f"  Response preview: {pr['response'][:150]}...")
        
        self.__class__.query_result = query_data
    
    def test_04_validate_debug_file(self, enable_debug_mode):
        """Validate the debug output file contains all required fields."""
        # Check for debug file
        debug_file = STORAGE_PATH / "topic_query.json"
        
        if not debug_file.exists():
            # Try topic-specific file
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"Test: {self.topic_name}"[:50])
            debug_file = STORAGE_PATH / f"topic_query_{safe_name}.json"
        
        assert debug_file.exists(), f"Debug file not found: {debug_file}"
        
        with open(debug_file) as f:
            debug_data = json.load(f)
        
        print(f"\n=== Debug File Validation ===")
        print(f"File: {debug_file}")
        
        # Validate required fields
        required_fields = [
            "topic_id",
            "topic_name",
            "question",
            "paper_responses",
            "aggregation_prompt",
            "final_answer",
            "model",
        ]
        
        for field in required_fields:
            assert field in debug_data, f"Debug file missing required field: {field}"
            print(f"✓ {field}: present")
        
        # Validate embedding fields
        embedding_fields = [
            ("topic_embedding_dim", "Topic embedding dimension"),
            ("question_embedding_dim", "Question embedding dimension"),
        ]
        
        for field, desc in embedding_fields:
            if field in debug_data:
                print(f"✓ {desc}: {debug_data[field]}")
                assert debug_data[field] > 0, f"{desc} should be > 0"
        
        # Validate papers_in_pool structure
        if "papers_in_pool" in debug_data:
            papers_pool = debug_data["papers_in_pool"]
            print(f"\n=== Papers in Pool ({len(papers_pool)}) ===")
            
            for i, paper in enumerate(papers_pool[:3]):
                print(f"\nPaper {i+1}:")
                print(f"  paper_id: {paper.get('paper_id')}")
                print(f"  arxiv_id: {paper.get('arxiv_id')}")
                print(f"  title: {paper.get('title', 'N/A')[:50]}...")
                print(f"  similarity_score: {paper.get('similarity_score')}")
                print(f"  embedding_dim: {paper.get('embedding_dim', 'N/A')}")
                
                assert "paper_id" in paper, "Paper missing paper_id"
                assert "similarity_score" in paper, "Paper missing similarity_score"
        
        # Validate paper_responses structure
        paper_responses = debug_data.get("paper_responses", [])
        print(f"\n=== Paper Responses ({len(paper_responses)}) ===")
        
        successful = [r for r in paper_responses if r.get("success")]
        failed = [r for r in paper_responses if not r.get("success")]
        
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            sample = successful[0]
            print(f"\nSample successful response:")
            print(f"  paper_id: {sample.get('paper_id')}")
            print(f"  arxiv_id: {sample.get('arxiv_id')}")
            print(f"  chunks_retrieved: {sample.get('chunks_retrieved', 'N/A')}")
            
            if sample.get("chunks"):
                print(f"  chunks: {len(sample['chunks'])} chunks")
                for j, chunk in enumerate(sample["chunks"][:2]):
                    print(f"    Chunk {j+1}: score={chunk.get('score')}, text={chunk.get('text', '')[:80]}...")
        
        # Validate aggregation prompt
        agg_prompt = debug_data.get("aggregation_prompt", "")
        print(f"\n=== Aggregation Prompt ===")
        print(f"Length: {len(agg_prompt)} chars")
        assert len(agg_prompt) > 100, "Aggregation prompt seems too short"
        assert "Question:" in agg_prompt, "Aggregation prompt missing question"
        assert "Paper" in agg_prompt, "Aggregation prompt missing paper references"
        
        # Validate final answer
        final_answer = debug_data.get("final_answer", "")
        print(f"\n=== Final Answer ===")
        print(f"Length: {len(final_answer)} chars")
        print(f"Preview: {final_answer[:200]}...")
        assert len(final_answer) > 50, "Final answer seems too short"
        
        print(f"\n✓ All debug validations passed!")
    
    def test_05_cleanup_topic(self, enable_debug_mode):
        """Clean up the test topic."""
        if not self.topic_id:
            pytest.skip("No topic to clean up")
        
        resp = requests.delete(f"{BASE_URL}/topic/{self.topic_id}")
        
        # Topic deletion should succeed or return 404 if already deleted
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
            json={"topic": topic, "limit": 10, "offset": 0}
        )
        
        if resp.status_code != 200:
            pytest.skip(f"Search failed: {resp.text}")
        
        data = resp.json()
        papers = data.get("papers", [])
        
        print(f"\n=== FP8 Topic Search ===")
        print(f"Topic: {topic}")
        print(f"Papers found: {len(papers)}")
        
        if papers:
            for i, paper in enumerate(papers[:5]):
                print(f"\n  {i+1}. {paper.get('title', 'N/A')[:60]}...")
                print(f"     Similarity: {paper.get('similarity_score', 'N/A')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
