# Validation Tests

Validation tests verify two things:
1. **Connectivity checks** - All required services are accessible
2. **Input validation** - API endpoints reject invalid inputs correctly (422 errors)

These tests are fast and should run before functional/integration tests to catch configuration issues early.

## Prerequisites

- Backend should be running (for connectivity checks)
- No LLM calls required (input validation uses TestClient without server exceptions)

## Running Tests

```bash
cd tests/validation
pytest -v

# Run only connectivity checks
pytest -v -k "connectivity"

# Run only input validation
pytest -v -k "input"
```

---

## Connectivity Checks

### test_connectivity_llm_endpoint

Verify LLM model endpoint is accessible.

Inputs:
- LLM base URL from config

Success criteria:
- HTTP GET to /v1/models returns status 200
- Response contains model list
- At least one model available

---

### test_connectivity_embedding_endpoint

Verify embedding model endpoint is accessible.

Inputs:
- Embedding base URL from config

Success criteria:
- HTTP GET to /v1/models returns status 200
- Response contains model list
- At least one embedding model available

---

### test_connectivity_backend

Verify backend API is accessible.

Inputs:
- BACKEND_URL (default: http://localhost:3100)

Success criteria:
- GET /health returns status 200
- Response body is {"status": "ok"}

---

### test_connectivity_frontend

Verify frontend is accessible.

Inputs:
- Frontend URL from config (default: http://localhost:3000)

Success criteria:
- HTTP GET to / returns status 200
- Response contains HTML content
- Page title contains expected text

---

### test_connectivity_database

Verify PostgreSQL database is accessible.

Inputs:
- Database connection params from environment (PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE)

Success criteria:
- Connection established successfully
- Simple query "SELECT 1" returns expected result
- pgvector extension is installed (SELECT * FROM pg_extension WHERE extname = 'vector')

---

### test_config_obtainable

Verify all configuration values are obtainable.

Inputs:
- None

Success criteria:
- GET /config returns status 200
- Response contains all expected config sections: endpoint, ui, classification, ingestion, topic_query
- No null values for required fields
- Schema metadata present for each setting

---

### test_config_ui_obtainable

Verify UI-specific config is obtainable.

Inputs:
- None

Success criteria:
- GET /ui-config returns status 200
- Response contains: hover_debounce_ms, max_similar_papers, tree_auto_save_interval_ms
- All values are valid integers > 0

---

## Input Validation Tests

These tests verify the API correctly rejects malformed or incomplete requests.

### test_input_arxiv_resolve_requires_identifier

POST /arxiv/resolve with empty body should return 400.

Expected error: "Provide arxiv_id or url"

---

### test_input_arxiv_download_requires_identifier

POST /arxiv/download with empty body should return 400.

Expected error: "Provide arxiv_id or url"

---

### test_input_pdf_extract_requires_path

POST /pdf/extract with empty body should return 422.

Expected: Validation error for missing pdf_path field

---

### test_input_summarize_structured_requires_pdf

POST /summarize/structured with empty body should return 422.

Expected: Validation error for missing pdf_path field

---

### test_input_embed_requires_text

POST /embed and POST /embed/abstract with empty body should return 422.

Expected: Validation error for missing text field

---

### test_input_embed_fulltext_requires_fields

POST /embed/fulltext with empty body should return 422.

Expected: Validation error for missing arxiv_id and pdf_path fields

---

### test_input_qa_requires_question

POST /qa with empty body should return 422.

Expected: Validation error for missing question field

---

### test_input_qa_structured_requires_arxiv_id

POST /qa/structured with empty body should return 422.

Expected: Validation error for missing arxiv_id field

---

### test_input_summary_merge_requires_fields

POST /summary/merge with empty body should return 422.

Expected: Validation error for missing arxiv_id and selected_qa fields

---

### test_input_summary_dedup_requires_arxiv_id

POST /summary/dedup with empty body should return 422.

Expected: Validation error for missing arxiv_id field

---

### test_input_classify_requires_fields

POST /classify with empty body should return 422.

Expected: Validation error for missing title and abstract fields

---

### test_input_abbreviate_requires_title

POST /abbreviate with empty body should return 422.

Expected: Validation error for missing title field

---

### test_input_reabbreviate_requires_arxiv_id

POST /papers/reabbreviate with empty body should return 422.

Expected: Validation error for missing arxiv_id field

---

### test_input_save_paper_requires_fields

POST /papers/save with empty body should return 422.

Expected: Validation error for missing arxiv_id, title, and authors fields

---

### test_input_batch_ingest_requires_source

POST /papers/batch-ingest with empty body should return 400.

Expected error: Must provide directory or slack_channel

---

### test_input_prefetch_requires_fields

POST /papers/prefetch with empty body should return 422.

Expected: Validation error for missing arxiv_id and title fields

---

### test_input_repo_search_requires_fields

POST /repos/search with empty body should return 422.

Expected: Validation error for missing arxiv_id and title fields

---

### test_input_references_fetch_requires_arxiv_id

POST /references/fetch with empty body should return 422.

Expected: Validation error for missing arxiv_id field

---

### test_input_references_explain_requires_fields

POST /references/explain with empty body should return 422.

Expected: Validation error for missing reference_id, source_paper_title, and cited_title fields

---

### test_input_similar_requires_arxiv_id

POST /papers/similar with empty body should return 422.

Expected: Validation error for missing arxiv_id field

---

### test_input_tree_node_requires_fields

POST /tree/node with empty body should return 422.

Expected: Validation error for missing node_id, name, and node_type fields

---

### test_input_topic_search_requires_topic

POST /topic/search with empty body should return 422.

Expected: Validation error for missing topic field

---

### test_input_topic_create_requires_fields

POST /topic/create with empty body should return 422.

Expected: Validation error for missing name and topic_query fields

---

### test_input_topic_query_requires_question

POST /topic/{topic_id}/query with empty body should return 422.

Expected: Validation error for missing question field
