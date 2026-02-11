"""Functional tests for paper operations using requests to call backend."""
import os
from pathlib import Path

import pytest
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3100")
TEST_STORAGE = Path(__file__).parent.parent / "storage"


class TestPaperDownload:
    """Test paper download functionality."""

    def test_arxiv_resolve(self, backend_available):
        """Resolve arXiv ID to metadata."""
        resp = requests.post(
            f"{BACKEND_URL}/arxiv/resolve",
            json={"arxiv_id": "1706.03762"},
            timeout=30
        )
        assert resp.status_code == 200
        data = resp.json()
        
        assert "title" in data
        assert "authors" in data
        assert "summary" in data
        assert "pdf_url" in data
        assert "Attention" in data["title"]

    def test_arxiv_download(self, backend_available, test_storage_dir):
        """Download paper PDF to test storage (scratch dir)."""
        download_dir = f"{test_storage_dir}/downloads/scratch"
        os.makedirs(download_dir, exist_ok=True)
        
        resp = requests.post(
            f"{BACKEND_URL}/arxiv/download",
            json={"arxiv_id": "1706.03762", "output_dir": download_dir},
            timeout=120
        )
        assert resp.status_code == 200
        data = resp.json()
        
        assert "pdf_path" in data
        pdf_path = Path(data["pdf_path"])
        assert pdf_path.exists()


class TestPdfExtraction:
    """Test PDF text extraction."""

    def test_pdf_extract(self, backend_available, test_storage_dir):
        """Extract text from PDF and verify quality."""
        pdf_path = f"{test_storage_dir}/downloads/local/1706.03762.pdf"
        assert Path(pdf_path).exists(), f"PDF not found at {pdf_path} - ensure test_arxiv_download runs first"
        
        resp = requests.post(
            f"{BACKEND_URL}/pdf/extract",
            json={"pdf_path": pdf_path},
            timeout=60
        )
        assert resp.status_code == 200, f"PDF extraction failed: {resp.text}"
        data = resp.json()
        
        assert "text" in data
        text = data["text"]
        
        # Basic length check
        assert len(text) > 5000, f"Extracted text too short: {len(text)} chars"
        
        # Quality checks - verify content is meaningful
        text_lower = text.lower()
        assert "attention" in text_lower, "Expected keyword 'attention' not found in extracted text"
        assert "transformer" in text_lower, "Expected keyword 'transformer' not found in extracted text"
        
        # Check for bad quality indicators
        lines = text.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]
        assert len(non_empty_lines) > 100, f"Too few non-empty lines: {len(non_empty_lines)}"
        
        # Check for malformed chunks (lines with mostly non-ASCII or garbage)
        garbage_lines = 0
        for line in non_empty_lines[:100]:
            if len(line) > 10:
                ascii_ratio = sum(1 for c in line if ord(c) < 128) / len(line)
                if ascii_ratio < 0.7:
                    garbage_lines += 1
        assert garbage_lines < 10, f"Too many garbage lines detected: {garbage_lines}"


class TestEmbedding:
    """Test embedding operations."""

    def test_embed_abstract(self, backend_available):
        """Generate embedding for abstract text."""
        resp = requests.post(
            f"{BACKEND_URL}/embed/abstract",
            json={"text": "The Transformer architecture uses attention mechanisms."},
            timeout=30
        )
        assert resp.status_code == 200, f"Embed abstract failed: {resp.text}"
        data = resp.json()
        
        assert "embedding" in data
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) > 0
        # Verify all values are floats
        assert all(isinstance(x, (int, float)) for x in data["embedding"][:10])

    def test_embed_chunks(self, backend_available):
        """Generate embeddings for multiple text chunks and verify each is non-empty."""
        chunks = [
            "The Transformer is a model architecture eschewing recurrence.",
            "Attention mechanisms allow modeling dependencies regardless of distance.",
            "Multi-head attention allows jointly attending to information from different positions."
        ]
        
        embeddings = []
        for i, chunk in enumerate(chunks):
            resp = requests.post(
                f"{BACKEND_URL}/embed/abstract",
                json={"text": chunk},
                timeout=30
            )
            assert resp.status_code == 200, f"Embed chunk {i} failed: {resp.text}"
            data = resp.json()
            
            assert "embedding" in data, f"Chunk {i} response missing embedding"
            emb = data["embedding"]
            assert isinstance(emb, list), f"Chunk {i} embedding not a list"
            assert len(emb) > 0, f"Chunk {i} embedding is empty"
            embeddings.append(emb)
        
        # Verify all embeddings have same dimension
        dims = [len(e) for e in embeddings]
        assert all(d == dims[0] for d in dims), f"Embedding dimensions don't match: {dims}"

    def test_embed_fulltext(self, backend_available, test_storage_dir):
        """Index full PDF text for PaperQA queries."""
        pdf_path = f"{test_storage_dir}/downloads/local/1706.03762.pdf"
        assert Path(pdf_path).exists(), f"PDF not found at {pdf_path}"
        
        resp = requests.post(
            f"{BACKEND_URL}/embed/fulltext",
            json={"pdf_path": pdf_path, "arxiv_id": "1706.03762"},
            timeout=300
        )
        assert resp.status_code == 200, f"Embed fulltext failed: {resp.text}"
        data = resp.json()
        
        assert "indexed" in data
        assert data["indexed"] is True


class TestSummarize:
    """Test summarization."""

    def test_summarize_with_pdf(self, backend_available, test_storage_dir):
        """Generate summary from PDF."""
        pdf_path = f"{test_storage_dir}/downloads/local/1706.03762.pdf"
        assert Path(pdf_path).exists(), f"PDF not found at {pdf_path}"
        
        resp = requests.post(
            f"{BACKEND_URL}/summarize",
            json={"pdf_path": pdf_path, "arxiv_id": "1706.03762"},
            timeout=300
        )
        
        assert resp.status_code == 200, f"Summarize failed: {resp.text}"
        data = resp.json()
        assert "summary" in data
        assert len(data["summary"]) > 50, "Summary too short"


class TestAbbreviate:
    """Test title abbreviation."""

    def test_abbreviate(self, backend_available):
        """Generate abbreviation for paper title."""
        resp = requests.post(
            f"{BACKEND_URL}/abbreviate",
            json={"title": "Attention Is All You Need"},
            timeout=60
        )
        
        assert resp.status_code == 200, f"Abbreviate failed: {resp.text}"
        data = resp.json()
        assert "abbreviation" in data


class TestClassify:
    """Test paper classification."""

    def test_classify(self, backend_available):
        """Classify paper into category."""
        resp = requests.post(
            f"{BACKEND_URL}/classify",
            json={
                "title": "Attention Is All You Need",
                "abstract": "We propose a network architecture, the Transformer.",
                "existing_categories": ["Computer Vision", "NLP", "RL"]
            },
            timeout=60
        )
        
        assert resp.status_code == 200, f"Classify failed: {resp.text}"
        data = resp.json()
        assert "category" in data


class TestQA:
    """Test Q&A functionality."""

    def test_qa_with_paper(self, backend_available, test_storage_dir):
        """Ask question about a paper."""
        pdf_path = f"{test_storage_dir}/downloads/local/1706.03762.pdf"
        assert Path(pdf_path).exists(), f"PDF not found at {pdf_path}"
        
        resp = requests.post(
            f"{BACKEND_URL}/qa",
            json={
                "arxiv_id": "1706.03762",
                "question": "What is the main contribution?",
                "pdf_path": pdf_path
            },
            timeout=180
        )
        
        assert resp.status_code == 200, f"QA failed: {resp.text}"
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 20, "Answer too short"


class TestReabbreviate:
    """Test paper reabbreviation."""

    def test_reabbreviate(self, backend_available, test_storage_dir):
        """Reabbreviate a paper - saves it first if needed."""
        arxiv_id = "1706.03762"
        pdf_path = f"{test_storage_dir}/downloads/local/{arxiv_id}.pdf"
        assert Path(pdf_path).exists(), f"PDF not found at {pdf_path}"
        
        # First get paper metadata
        resolve_resp = requests.post(
            f"{BACKEND_URL}/arxiv/resolve",
            json={"arxiv_id": arxiv_id},
            timeout=30
        )
        assert resolve_resp.status_code == 200
        metadata = resolve_resp.json()
        
        # Save paper to database
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
        
        # Paper might already exist (409) or be saved successfully (200)
        assert save_resp.status_code in [200, 409], f"Save failed: {save_resp.text}"
        
        # Now reabbreviate
        resp = requests.post(
            f"{BACKEND_URL}/papers/reabbreviate",
            json={"arxiv_id": arxiv_id},
            timeout=60
        )
        
        assert resp.status_code == 200, f"Reabbreviate failed: {resp.text}"
