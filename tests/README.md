# Test Suite Documentation

This document summarizes what each test file does and indicates whether mock data is used.

## Test Configuration

Tests are configured in `conftest.py`:
- Uses FastAPI TestClient with `raise_server_exceptions=False` to capture HTTP error responses
- Provides shared fixtures for sample arXiv data (paper: "Attention Is All You Need", ID: 1706.03762)

## Test Markers

- `@pytest.mark.slow` - Tests that download real files or take significant time
- `@pytest.mark.external` - Tests that require external LLM/embedding endpoints

## Test Files

### test_health.py

| Test | Description | Mock Data |
|------|-------------|-----------|
| `test_health` | Verifies `/health` endpoint returns `{"status": "ok"}` | No (real endpoint, no external deps) |

---

### test_arxiv.py

| Test | Description | Mock Data |
|------|-------------|-----------|
| `test_arxiv_resolve_with_id` | Resolves paper metadata by arXiv ID, verifies "Attention Is All You Need" paper | No (real arXiv API) |
| `test_arxiv_resolve_with_url` | Resolves paper metadata by arXiv URL | No (real arXiv API) |
| `test_arxiv_resolve_no_identifier` | Verifies 400 error when no ID/URL provided | No |
| `test_arxiv_resolve_invalid_id` | Verifies error with invalid arXiv ID | No (real arXiv API) |
| `test_arxiv_download` | Downloads actual PDF from arXiv to temp directory (marked `@slow`) | No (real arXiv download) |

---

### test_embed.py

| Test | Description | Mock Data |
|------|-------------|-----------|
| `test_embed` | Generates embedding for sample text (marked `@external`) | No (real embedding endpoint) |

**Note**: Skips if embedding endpoint unavailable. Set `REQUIRE_EXTERNAL_ENDPOINTS=1` to fail instead.

---

### test_pdf_extract.py

| Test | Description | Mock Data |
|------|-------------|-----------|
| `test_pdf_extract_file_not_found` | Verifies 404 error for nonexistent PDF path | No |
| `test_pdf_extract_with_real_pdf` | Downloads real PDF then extracts text using PaperQA2 (marked `@slow`) | No (real PDF, real extraction) |

---

### test_qa.py

| Test | Description | Mock Data |
|------|-------------|-----------|
| `test_qa_with_context` | Answers question using provided context about Transformers (marked `@external`) | **Yes** - hardcoded context about Transformers |
| `test_qa_with_pdf` | Downloads real PDF, answers question about it (marked `@external`, `@slow`) | No (real PDF, real LLM) |
| `test_qa_no_input` | Verifies error when no context/pdf_path provided | No |

---

### test_summarize.py

| Test | Description | Mock Data |
|------|-------------|-----------|
| `test_summarize_with_text` | Summarizes provided sample text about attention mechanisms (marked `@external`) | **Yes** - hardcoded text about attention |
| `test_summarize_with_pdf` | Downloads real PDF and summarizes it (marked `@external`, `@slow`) | No (real PDF, real LLM) |
| `test_summarize_no_input` | Verifies error when no input provided | No |

---

## Running Tests

```bash
# Run all tests
make test

# Run in venv manually
source .venv/bin/activate
pytest tests/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# Skip tests requiring external endpoints
pytest tests/ -v -m "not external"

# Fail on missing external endpoints (CI mode)
REQUIRE_EXTERNAL_ENDPOINTS=1 pytest tests/ -v
```

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| Total tests | 15 | |
| Tests using mock data | 2 | `test_qa_with_context`, `test_summarize_with_text` use hardcoded sample text |
| Tests hitting real APIs | 13 | arXiv API, LLM endpoints, embedding endpoints |
| Slow tests | 4 | Download real PDFs from arXiv |
| External endpoint tests | 5 | Require LLM/embedding endpoints to be running |
