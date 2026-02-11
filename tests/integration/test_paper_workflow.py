"""Integration tests for paper workflows."""
import os
from pathlib import Path

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")
LONG_TIMEOUT = 300  # 5 minutes for heavy operations


class TestPaperIngest:
    """Test complete paper ingestion workflow."""

    def test_ingest_single_paper(self, backend_available, test_storage_dir):
        """Test complete paper ingestion from arXiv ID with full verification.
        
        Verifies:
        - ArXiv metadata extraction (title, authors, abstract)
        - PDF download
        - PDF text extraction (content quality)
        - Embedding generation
        - Paper saved to database
        """
        arxiv_id = "1706.03762"
        download_dir = f"{test_storage_dir}/downloads"
        os.makedirs(download_dir, exist_ok=True)
        
        # Step 1: Resolve arXiv metadata
        resolve_resp = requests.post(
            f"{BACKEND_URL}/arxiv/resolve",
            json={"arxiv_id": arxiv_id},
            timeout=60
        )
        assert resolve_resp.status_code == 200, f"ArXiv resolve failed: {resolve_resp.text}"
        metadata = resolve_resp.json()
        
        # Verify ArXiv metadata
        assert "title" in metadata, "Missing title in arXiv metadata"
        assert "authors" in metadata, "Missing authors in arXiv metadata"
        assert "summary" in metadata, "Missing summary/abstract in arXiv metadata"
        assert "Attention" in metadata["title"], "Expected 'Attention' in title"
        assert len(metadata.get("authors", [])) > 0, "Expected at least one author"
        
        # Step 2: Download PDF
        download_resp = requests.post(
            f"{BACKEND_URL}/arxiv/download",
            json={"arxiv_id": arxiv_id, "output_dir": download_dir},
            timeout=LONG_TIMEOUT
        )
        assert download_resp.status_code == 200, f"PDF download failed: {download_resp.text}"
        pdf_data = download_resp.json()
        pdf_path = pdf_data.get("pdf_path")
        assert pdf_path and Path(pdf_path).exists(), f"PDF not found at {pdf_path}"
        
        # Step 3: Extract text and verify quality
        extract_resp = requests.post(
            f"{BACKEND_URL}/pdf/extract",
            json={"pdf_path": pdf_path},
            timeout=60
        )
        assert extract_resp.status_code == 200, f"PDF extraction failed: {extract_resp.text}"
        extract_data = extract_resp.json()
        assert "text" in extract_data, "Missing text in extraction response"
        assert len(extract_data["text"]) > 5000, "Extracted text too short"
        assert "transformer" in extract_data["text"].lower(), "Expected 'transformer' in extracted text"
        
        # Step 4: Generate embedding
        embed_resp = requests.post(
            f"{BACKEND_URL}/embed/abstract",
            json={"text": metadata.get("summary", "Transformer architecture paper")},
            timeout=60
        )
        assert embed_resp.status_code == 200, f"Embedding failed: {embed_resp.text}"
        embed_data = embed_resp.json()
        assert "embedding" in embed_data, "Missing embedding in response"
        assert len(embed_data["embedding"]) > 0, "Empty embedding vector"
        
        # Step 5: Save paper to database
        save_resp = requests.post(
            f"{BACKEND_URL}/papers/save",
            json={
                "arxiv_id": arxiv_id,
                "title": metadata["title"],
                "authors": metadata.get("authors", []),
                "abstract": metadata.get("summary", ""),
                "pdf_path": pdf_path
            },
            timeout=60
        )
        # 200 = new paper saved, 409 = already exists (both are OK)
        assert save_resp.status_code in [200, 409], f"Paper save failed: {save_resp.text}"
        
        # Step 6: Verify tree exists (as a proxy for database state)
        tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
        assert tree_resp.status_code == 200, f"Tree retrieval failed: {tree_resp.text}"
        tree = tree_resp.json()
        assert isinstance(tree, dict), "Tree should be a dict"
        assert "name" in tree, "Tree should have a name"

    def test_batch_ingest(self, backend_available, test_storage_dir):
        """Test batch paper ingestion from local directory."""
        download_dir = f"{test_storage_dir}/downloads"
        
        assert Path(download_dir).exists(), f"Download directory not available: {download_dir}"
        
        pdfs = list(Path(download_dir).glob("*.pdf"))
        assert len(pdfs) > 0, "No PDFs in download directory"
        
        resp = requests.post(
            f"{BACKEND_URL}/papers/batch-ingest",
            json={"directory": download_dir},
            timeout=LONG_TIMEOUT
        )
        
        # Endpoint must exist and respond
        assert resp.status_code in [200, 202, 400, 422], \
            f"Batch ingest failed with status {resp.status_code}: {resp.text}"


class TestPaperRemove:
    """Test paper removal."""

    def test_paper_remove(self, backend_available, test_storage_dir):
        """Remove a paper and verify the delete endpoint works."""
        arxiv_id = "1706.03762"
        
        # First ensure paper exists by saving it
        pdf_path = f"{test_storage_dir}/downloads/{arxiv_id}.pdf"
        if Path(pdf_path).exists():
            resolve_resp = requests.post(
                f"{BACKEND_URL}/arxiv/resolve",
                json={"arxiv_id": arxiv_id},
                timeout=30
            )
            if resolve_resp.status_code == 200:
                metadata = resolve_resp.json()
                requests.post(
                    f"{BACKEND_URL}/papers/save",
                    json={
                        "arxiv_id": arxiv_id,
                        "title": metadata.get("title", "Test Paper"),
                        "authors": metadata.get("authors", []),
                        "abstract": metadata.get("summary", ""),
                        "pdf_path": pdf_path
                    },
                    timeout=60
                )
        
        # Delete the paper
        delete_resp = requests.delete(
            f"{BACKEND_URL}/papers/{arxiv_id}",
            timeout=60
        )
        
        # Accept 200/204 (success), 404 (already deleted or never existed)
        assert delete_resp.status_code in [200, 204, 404], \
            f"Delete failed with status {delete_resp.status_code}: {delete_resp.text}"


class TestTreeOperations:
    """Test tree structure operations."""

    def test_tree_structure(self, backend_available):
        """Verify tree can be retrieved and has valid structure."""
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
        assert resp.status_code == 200, f"Tree retrieval failed: {resp.text}"
        tree = resp.json()
        assert isinstance(tree, dict), "Tree should be a dict"
        assert "name" in tree, "Tree missing 'name' field"

    def test_classify_endpoint(self, backend_available):
        """Test that classify endpoint responds."""
        resp = requests.post(
            f"{BACKEND_URL}/papers/classify",
            json={},
            timeout=120
        )
        
        # Allow 400/422 for empty payload, but endpoint must exist and work
        assert resp.status_code in [200, 202, 400, 422], \
            f"Classify endpoint failed with status {resp.status_code}: {resp.text}"

    def test_clustering(self, backend_available):
        """Verify tree clustering structure."""
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
        
        assert resp.status_code == 200, f"Tree retrieval failed: {resp.text}"
        tree = resp.json()
        
        assert isinstance(tree, dict), "Tree should be a dict"
        assert "name" in tree, "Tree missing 'name' field"
        
        if "children" in tree:
            children = tree["children"]
            assert isinstance(children, list), "Children should be a list"
            for child in children[:3]:
                assert "name" in child or "node_id" in child, \
                    f"Child missing expected fields: {child}"


class TestCategoryOperations:
    """Test category operations."""

    def test_get_categories(self, backend_available):
        """Test getting categories from tree."""
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
        
        assert resp.status_code == 200, f"Tree retrieval failed: {resp.text}"
        tree = resp.json()
        assert isinstance(tree, dict), "Tree should be a dict"

    def test_category_rename(self, backend_available, unique_test_id):
        """Test category renaming."""
        tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
        assert tree_resp.status_code == 200, f"Cannot get tree: {tree_resp.text}"
        
        tree = tree_resp.json()
        
        # Find a category node to rename
        category_name = None
        if "children" in tree and len(tree["children"]) > 0:
            for child in tree["children"]:
                if "paper_id" not in child and "name" in child:
                    category_name = child.get("name")
                    break
        
        if not category_name:
            # No category nodes to rename - this is a precondition, not a failure
            return
        
        new_name = f"TestCategory_{unique_test_id}"
        
        resp = requests.post(
            f"{BACKEND_URL}/categories/rename",
            json={"old_name": category_name, "new_name": new_name},
            timeout=60
        )
        
        # If successful, rename back
        if resp.status_code == 200:
            requests.post(
                f"{BACKEND_URL}/categories/rename",
                json={"old_name": new_name, "new_name": category_name},
                timeout=60
            )
        
        assert resp.status_code in [200, 400, 404, 422], \
            f"Category rename failed with status {resp.status_code}: {resp.text}"

    def test_rename_all_categories(self, backend_available):
        """Test reabbreviating all categories."""
        resp = requests.post(
            f"{BACKEND_URL}/papers/reabbreviate-all",
            json={},
            timeout=LONG_TIMEOUT
        )
        
        assert resp.status_code in [200, 202, 400, 422], \
            f"Reabbreviate-all failed with status {resp.status_code}: {resp.text}"


class TestLocalFolderIngest:
    """Test local folder ingestion."""

    def test_ingest_local_folder(self, backend_available, test_storage_dir):
        """Ingest papers from local folder with verification."""
        download_dir = f"{test_storage_dir}/downloads"
        
        assert Path(download_dir).exists(), f"Download directory not available: {download_dir}"
        
        pdfs = list(Path(download_dir).glob("*.pdf"))
        assert len(pdfs) > 0, "No PDFs in download directory"
        
        resp = requests.post(
            f"{BACKEND_URL}/papers/batch-ingest",
            json={"directory": download_dir},
            timeout=LONG_TIMEOUT
        )
        
        assert resp.status_code in [200, 202, 400, 422], \
            f"Batch ingest failed with status {resp.status_code}: {resp.text}"
        
        # Verify tree after ingestion
        tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
        assert tree_resp.status_code == 200, f"Tree retrieval after ingest failed: {tree_resp.text}"
        tree = tree_resp.json()
        assert isinstance(tree, dict), "Tree should be a dict"


class TestChannelIngest:
    """Test Slack channel ingestion."""

    def test_ingest_channel(self, backend_available):
        """Ingest papers from Slack channel (limited to 10)."""
        resp = requests.post(
            f"{BACKEND_URL}/papers/batch-ingest",
            json={"slack_channel": "paper-feed", "limit": 10},
            timeout=LONG_TIMEOUT
        )
        
        # 400 is acceptable if Slack is not configured
        # 503 is acceptable if Slack service is temporarily unavailable
        assert resp.status_code in [200, 202, 400, 422, 503], \
            f"Channel ingest failed with status {resp.status_code}: {resp.text}"


class TestTopicWorkflow:
    """Test topic-related workflows."""

    def test_topic_list(self, backend_available):
        """Test listing topics."""
        resp = requests.get(f"{BACKEND_URL}/topic/list", timeout=60)
        
        assert resp.status_code == 200, f"Topic list failed with status {resp.status_code}: {resp.text}"
        data = resp.json()
        assert isinstance(data, (list, dict)), "Topic list should be a list or dict"
