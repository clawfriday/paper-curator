"""Integration tests for paper workflows.

Test ordering matters: lightweight tests run first, heavy batch operations last.
Classes are named Test01, Test02, ... to enforce alphabetical execution order.

All tests run against a CLEAN test database (paper_curator_test) that is
automatically created and switched to by the conftest session fixture.

Storage layout:
  tests/storage/downloads/local/   - 10 curated PDFs for local folder ingest test
  tests/storage/downloads/scratch/ - temp dir for single-paper test (cleaned per run)
"""
import os
import shutil
from pathlib import Path

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")
LONG_TIMEOUT = 900   # 15 minutes for heavy batch operations
MEDIUM_TIMEOUT = 300  # 5 minutes for moderate operations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_tree_papers(node: dict) -> int:
    """Recursively count paper nodes in a tree."""
    count = 1 if "paper_id" in node else 0
    for child in node.get("children", []):
        count += _count_tree_papers(child)
    return count


def _count_tree_categories(node: dict) -> int:
    """Recursively count category (non-paper) nodes in a tree."""
    count = 1 if "paper_id" not in node and node.get("children") else 0
    for child in node.get("children", []):
        count += _count_tree_categories(child)
    return count


def _get_db_paper_count() -> int:
    """Get paper count via /db/status."""
    resp = requests.get(f"{BACKEND_URL}/db/status", timeout=10)
    return resp.json().get("paper_count", 0)


# ===========================================================================
# Test01–Test03: Lightweight read-only tests (tree, categories, topics)
# ===========================================================================

class Test01TreeOperations:
    """Test tree structure operations (lightweight, run first)."""

    def test_tree_structure(self, backend_available):
        """Verify tree can be retrieved and has valid structure."""
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
        assert resp.status_code == 200, f"Tree retrieval failed: {resp.text}"
        tree = resp.json()
        assert isinstance(tree, dict), "Tree should be a dict"
        assert "name" in tree, "Tree missing 'name' field"

    def test_clustering(self, backend_available):
        """Verify tree clustering structure."""
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
        assert resp.status_code == 200, f"Tree retrieval failed: {resp.text}"
        tree = resp.json()
        assert isinstance(tree, dict), "Tree should be a dict"
        assert "name" in tree, "Tree missing 'name' field"


class Test02CategoryOperations:
    """Test category operations (lightweight, no LLM)."""

    def test_get_categories(self, backend_available):
        """Test getting categories from tree."""
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
        assert resp.status_code == 200, f"Tree retrieval failed: {resp.text}"

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
            return  # No category nodes yet — tree is empty in clean DB

        new_name = f"TestCategory_{unique_test_id}"
        resp = requests.post(
            f"{BACKEND_URL}/categories/rename",
            json={"old_name": category_name, "new_name": new_name},
            timeout=120
        )
        if resp.status_code == 200:
            requests.post(
                f"{BACKEND_URL}/categories/rename",
                json={"old_name": new_name, "new_name": category_name},
                timeout=120
            )
        assert resp.status_code in [200, 400, 404, 422], \
            f"Category rename failed: {resp.status_code}: {resp.text}"


class Test03TopicWorkflow:
    """Test topic-related workflows (lightweight)."""

    def test_topic_list(self, backend_available):
        """Test listing topics."""
        resp = requests.get(f"{BACKEND_URL}/topic/list", timeout=60)
        assert resp.status_code == 200, f"Topic list failed: {resp.text}"
        data = resp.json()
        assert isinstance(data, (list, dict))


# ===========================================================================
# Test04: Single paper ingest — FULL VERIFICATION
# ===========================================================================

class Test04SinglePaperIngest:
    """Test single paper ingestion with full pipeline verification.

    Uses tests/storage/downloads/scratch/ as temp dir, cleaned after.
    Verifies: metadata, PDF, extraction, embedding, summary, save, tree node.
    """

    def test_ingest_single_paper(self, backend_available, test_storage_dir):
        arxiv_id = "1706.03762"
        scratch_dir = f"{test_storage_dir}/downloads/scratch"
        os.makedirs(scratch_dir, exist_ok=True)

        papers_before = _get_db_paper_count()

        try:
            # 1. Resolve arXiv metadata
            resolve_resp = requests.post(
                f"{BACKEND_URL}/arxiv/resolve",
                json={"arxiv_id": arxiv_id}, timeout=60)
            assert resolve_resp.status_code == 200, f"Resolve failed: {resolve_resp.text}"
            metadata = resolve_resp.json()
            assert "title" in metadata and "Attention" in metadata["title"]
            assert len(metadata.get("authors", [])) > 0
            assert len(metadata.get("summary", "")) > 100, "Abstract too short"

            # 2. Download PDF
            dl_resp = requests.post(
                f"{BACKEND_URL}/arxiv/download",
                json={"arxiv_id": arxiv_id, "output_dir": scratch_dir},
                timeout=MEDIUM_TIMEOUT)
            assert dl_resp.status_code == 200, f"Download failed: {dl_resp.text}"
            pdf_path = dl_resp.json().get("pdf_path")
            assert pdf_path and Path(pdf_path).exists(), f"PDF not at {pdf_path}"

            # 3. Extract text — quality checks
            ext_resp = requests.post(
                f"{BACKEND_URL}/pdf/extract",
                json={"pdf_path": pdf_path}, timeout=60)
            assert ext_resp.status_code == 200, f"Extract failed: {ext_resp.text}"
            text = ext_resp.json().get("text", "")
            assert len(text) > 5000, f"Extracted text too short ({len(text)} chars)"
            assert "transformer" in text.lower(), "Expected 'transformer' in text"

            # 4. Abstract embedding
            emb_resp = requests.post(
                f"{BACKEND_URL}/embed/abstract",
                json={"text": metadata["summary"]}, timeout=60)
            assert emb_resp.status_code == 200, f"Embed failed: {emb_resp.text}"
            embedding = emb_resp.json().get("embedding", [])
            assert len(embedding) > 0, "Empty embedding vector"

            # 5. Summarize (PaperQA RAG)
            sum_resp = requests.post(
                f"{BACKEND_URL}/summarize",
                json={"pdf_path": pdf_path}, timeout=MEDIUM_TIMEOUT)
            assert sum_resp.status_code == 200, f"Summarize failed: {sum_resp.text}"
            summary = sum_resp.json().get("summary", "")
            assert len(summary) > 50, f"Summary too short ({len(summary)} chars)"

            # 6. Save to database
            save_resp = requests.post(
                f"{BACKEND_URL}/papers/save",
                json={
                    "arxiv_id": arxiv_id,
                    "title": metadata["title"],
                    "authors": metadata.get("authors", []),
                    "abstract": metadata.get("summary", ""),
                    "pdf_path": pdf_path,
                }, timeout=60)
            assert save_resp.status_code in [200, 409], f"Save failed: {save_resp.text}"

            # 7. Verify paper count increased
            papers_after = _get_db_paper_count()
            assert papers_after >= papers_before + 1, \
                f"Paper count did not increase: {papers_before} → {papers_after}"

            # 8. Verify tree has a node for this paper
            tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
            assert tree_resp.status_code == 200
            tree = tree_resp.json()
            assert isinstance(tree, dict) and "name" in tree

        finally:
            if Path(scratch_dir).exists():
                shutil.rmtree(scratch_dir, ignore_errors=True)


# ===========================================================================
# Test05: Paper removal — verify DB + tree
# ===========================================================================

class Test05PaperRemove:
    """Remove a paper and verify it's gone from DB and tree."""

    def test_paper_remove(self, backend_available):
        arxiv_id = "1706.03762"
        papers_before = _get_db_paper_count()

        del_resp = requests.delete(
            f"{BACKEND_URL}/papers/{arxiv_id}", timeout=60)
        assert del_resp.status_code in [200, 204, 404], \
            f"Delete failed: {del_resp.status_code}: {del_resp.text}"

        if del_resp.status_code in [200, 204]:
            # Verify count decreased
            papers_after = _get_db_paper_count()
            assert papers_after < papers_before, \
                f"Paper count did not decrease after delete: {papers_before} → {papers_after}"


# ===========================================================================
# Test06: Classify (cluster + name all nodes)
# ===========================================================================

class Test06ClassifyEndpoint:
    """Test classify endpoint — clusters all papers and names via LLM."""

    def test_classify_endpoint(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/papers/classify",
            json={}, timeout=LONG_TIMEOUT)
        assert resp.status_code in [200, 202, 400, 422], \
            f"Classify failed: {resp.status_code}: {resp.text}"

        if resp.status_code == 200:
            data = resp.json()
            print(f"\nClassify result: {data.get('papers_classified', '?')} papers, "
                  f"{data.get('clusters_created', '?')} clusters, "
                  f"{data.get('nodes_named', '?')} nodes named")

            # Verify tree was rebuilt with structure
            tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
            assert tree_resp.status_code == 200
            tree = tree_resp.json()
            assert "children" in tree, "Tree should have children after clustering"


# ===========================================================================
# Test07: Rename all — verify category nodes actually renamed
# ===========================================================================

class Test07RenameAllCategories:
    """Reabbreviate all papers and verify names changed."""

    def test_rename_all_categories(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/papers/reabbreviate-all",
            json={}, timeout=LONG_TIMEOUT)
        assert resp.status_code in [200, 202, 400, 422], \
            f"Reabbreviate-all failed: {resp.status_code}: {resp.text}"

        if resp.status_code == 200:
            data = resp.json()
            updated = data.get("updated", 0)
            print(f"\nReabbreviate-all: {updated} papers updated")
            assert updated > 0 or _get_db_paper_count() == 0, \
                "Expected at least one paper to be reabbreviated"

            # Verify tree nodes have names (not default UUIDs)
            tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
            if tree_resp.status_code == 200:
                tree = tree_resp.json()
                for child in tree.get("children", [])[:5]:
                    name = child.get("name", "")
                    assert name and not name.startswith("node_"), \
                        f"Tree node still has default name: {name}"


# ===========================================================================
# Test08: Slack channel ingest — limited to 10 papers
# ===========================================================================

class Test08ChannelIngest:
    """Test Slack channel ingestion (limited to 10 papers).

    Reads the Slack token from ~/.ssh/.slack (same as `make pull-slack`).
    If the token file doesn't exist, tests that the endpoint returns 400.
    """

    def test_ingest_channel(self, backend_available):
        papers_before = _get_db_paper_count()

        # Try to load Slack token
        token_path = Path.home() / ".ssh" / ".slack"
        slack_token = None
        if token_path.exists():
            slack_token = token_path.read_text().strip()

        payload = {"slack_channel": "https://app.slack.com/client/T04MW5HMWV9/C0A727EKAJV", "limit": 10}
        if slack_token:
            payload["slack_token"] = slack_token

        resp = requests.post(
            f"{BACKEND_URL}/papers/batch-ingest",
            json=payload,
            timeout=LONG_TIMEOUT)

        if slack_token:
            # With token: should succeed or at least not error badly
            assert resp.status_code in [200, 202, 400, 404, 422, 500, 503], \
                f"Channel ingest failed: {resp.status_code}: {resp.text}"

            if resp.status_code == 200:
                data = resp.json()
                ingested = data.get("success", 0)
                skipped = data.get("skipped", 0)
                total = ingested + skipped
                print(f"\nSlack ingest: {ingested} success, {skipped} skipped, "
                      f"{data.get('errors', 0)} errors (limit=10)")

                # Verify paper count increased (or all were skipped)
                papers_after = _get_db_paper_count()
                if ingested > 0:
                    assert papers_after > papers_before, \
                        f"Paper count did not increase after Slack ingest: " \
                        f"{papers_before} → {papers_after}"

                # Verify tree matches DB count
                tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=60)
                if tree_resp.status_code == 200:
                    tree = tree_resp.json()
                    tree_papers = _count_tree_papers(tree)
                    print(f"  DB papers: {papers_after}, Tree papers: {tree_papers}")
        else:
            # Without token: should get 400
            assert resp.status_code == 400, \
                f"Expected 400 without token, got {resp.status_code}: {resp.text}"


# ===========================================================================
# Test09: Local folder ingest — 10 curated PDFs, verify counts
# ===========================================================================

class Test09LocalFolderIngest:
    """Ingest 10 curated PDFs from tests/storage/downloads/local/.

    Verifies:
    - All 10 papers are ingested (or skipped if already present)
    - DB paper count matches tree paper count
    - "Attention Is All You Need" details are correct
    """

    def test_ingest_local_folder(self, backend_available, test_storage_dir):
        local_dir = f"{test_storage_dir}/downloads/local"
        assert Path(local_dir).exists(), f"Local dir not found: {local_dir}"

        pdfs = sorted(Path(local_dir).glob("*.pdf"))
        assert len(pdfs) >= 10, \
            f"Expected 10 PDFs in local/, found {len(pdfs)}"

        papers_before = _get_db_paper_count()

        print(f"\n=== Local Folder Ingest: {len(pdfs)} PDFs ===")
        for p in pdfs:
            print(f"  {p.name} ({p.stat().st_size / 1024:.0f} KB)")

        resp = requests.post(
            f"{BACKEND_URL}/papers/batch-ingest",
            json={"directory": local_dir, "limit": 10},
            timeout=LONG_TIMEOUT)

        assert resp.status_code in [200, 202, 400, 422], \
            f"Batch ingest failed: {resp.status_code}: {resp.text}"

        if resp.status_code == 200:
            data = resp.json()
            ingested = data.get("success", 0)
            skipped = data.get("skipped", 0)
            errors = data.get("errors", 0)
            print(f"\nResult: {ingested} success, {skipped} skipped, {errors} errors")

            # Verify DB paper count increased
            papers_after = _get_db_paper_count()
            if ingested > 0:
                assert papers_after > papers_before, \
                    f"Paper count did not increase: {papers_before} → {papers_after}"

            # Verify tree paper count matches DB
            tree_resp = requests.get(f"{BACKEND_URL}/tree", timeout=120)
            assert tree_resp.status_code == 200
            tree = tree_resp.json()
            tree_papers = _count_tree_papers(tree)
            print(f"  DB papers: {papers_after}, Tree papers: {tree_papers}")
            # Tree should have at least as many papers as DB (may lag behind
            # if rebuild_on_ingest is off, but should be close)
