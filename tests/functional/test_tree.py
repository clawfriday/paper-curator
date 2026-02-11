"""Functional tests for tree structure operations."""
import os
import requests
from pathlib import Path

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")


class TestTreeStructure:
    """Test tree structure functionality."""

    def test_get_tree(self, backend_available):
        """Retrieve full tree structure."""
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=30)
        assert resp.status_code == 200
        tree = resp.json()
        assert isinstance(tree, dict)
        assert "name" in tree

    def test_tree_has_structure(self, backend_available):
        """Verify tree has expected structure."""
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=30)
        assert resp.status_code == 200
        tree = resp.json()
        
        assert "name" in tree
        if "children" in tree:
            assert isinstance(tree["children"], list)
            if len(tree["children"]) > 0:
                child = tree["children"][0]
                assert "name" in child or "node_id" in child


class TestStructuredAnalysis:
    """Test structured analysis functionality."""

    def test_structured_summary(self, backend_available, test_storage_dir):
        """Generate structured summary."""
        pdf_path = f"{test_storage_dir}/downloads/1706.03762.pdf"
        assert Path(pdf_path).exists(), f"PDF not found at {pdf_path} - ensure download runs first"
        
        # Long timeout but let it fail if it times out
        resp = requests.post(
            f"{BACKEND_URL}/summarize/structured",
            json={"pdf_path": pdf_path, "arxiv_id": "1706.03762"},
            timeout=600  # 10 minutes for complex operations
        )
        
        assert resp.status_code == 200, f"Structured summary failed: {resp.text}"
        data = resp.json()
        
        # The structured endpoint returns 'components' and 'sections'
        has_content = (
            "summary" in data or 
            "structured_summary" in data or 
            "result" in data or
            "components" in data or
            "sections" in data
        )
        assert has_content, f"Response missing expected content fields: {list(data.keys())}"
        
        # Get full response content for validation
        full_content = str(data)
        assert len(full_content) > 500, f"Response too short: {len(full_content)} chars"
        
        # LLM validation: check for Transformer-related content
        content_lower = full_content.lower()
        has_relevant_content = any(word in content_lower for word in [
            "attention", "transformer", "encoder", "decoder", "layer"
        ])
        assert has_relevant_content, f"Summary doesn't mention expected concepts in: {full_content[:300]}"

    def test_qa_structured(self, backend_available, test_storage_dir):
        """Test structured Q&A with LLM validation."""
        pdf_path = f"{test_storage_dir}/downloads/1706.03762.pdf"
        assert Path(pdf_path).exists(), f"PDF not found at {pdf_path}"
        
        resp = requests.post(
            f"{BACKEND_URL}/qa/structured",
            json={
                "arxiv_id": "1706.03762",
                "question": "What are the key components of the Transformer architecture?",
                "pdf_path": pdf_path
            },
            timeout=300
        )
        
        assert resp.status_code == 200, f"Structured QA failed: {resp.text}"
        data = resp.json()
        
        # The structured QA endpoint returns 'components' and 'sections'
        has_content = (
            "answer" in data or 
            "response" in data or 
            "result" in data or
            "components" in data or
            "sections" in data
        )
        assert has_content, f"Response missing expected content fields: {list(data.keys())}"
        
        # Get full response content for validation
        full_content = str(data)
        assert len(full_content) > 500, f"Response too short: {len(full_content)} chars"
        
        # Check for relevant keywords in the full response
        content_lower = full_content.lower()
        has_relevant_content = any(word in content_lower for word in [
            "attention", "encoder", "decoder", "layer", "head", "transformer"
        ])
        assert has_relevant_content, f"Answer doesn't contain expected Transformer concepts in: {full_content[:300]}"
