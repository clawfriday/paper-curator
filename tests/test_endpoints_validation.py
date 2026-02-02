"""Validation tests for key endpoints."""
import os

import pytest


def test_config_endpoint(client):
    """GET /config should return UI config keys."""
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "hover_debounce_ms" in data
    assert "max_similar_papers" in data
    assert "tree_auto_save_interval_ms" in data


def test_arxiv_download_requires_identifier(client):
    """POST /arxiv/download requires arxiv_id or url."""
    response = client.post("/arxiv/download", json={})
    assert response.status_code == 400
    assert "Provide arxiv_id or url" in response.json()["detail"]


def test_pdf_extract_requires_path(client):
    """POST /pdf/extract requires pdf_path."""
    response = client.post("/pdf/extract", json={})
    assert response.status_code == 422


def test_summarize_structured_requires_pdf(client):
    """POST /summarize/structured requires pdf_path."""
    response = client.post("/summarize/structured", json={})
    assert response.status_code == 422


def test_embed_requires_text(client):
    """POST /embed and /embed/abstract require text."""
    response = client.post("/embed", json={})
    assert response.status_code == 422
    response = client.post("/embed/abstract", json={})
    assert response.status_code == 422


def test_embed_fulltext_requires_fields(client):
    """POST /embed/fulltext requires arxiv_id and pdf_path."""
    response = client.post("/embed/fulltext", json={})
    assert response.status_code == 422


def test_qa_requires_question(client):
    """POST /qa requires question."""
    response = client.post("/qa", json={})
    assert response.status_code == 422


def test_qa_structured_requires_arxiv_id(client):
    """POST /qa/structured requires arxiv_id."""
    response = client.post("/qa/structured", json={})
    assert response.status_code == 422


def test_summary_merge_requires_fields(client):
    """POST /summary/merge requires arxiv_id and selected_qa."""
    response = client.post("/summary/merge", json={})
    assert response.status_code == 422


def test_summary_dedup_requires_arxiv_id(client):
    """POST /summary/dedup requires arxiv_id."""
    response = client.post("/summary/dedup", json={})
    assert response.status_code == 422


def test_classify_requires_fields(client):
    """POST /classify requires title and abstract."""
    response = client.post("/classify", json={})
    assert response.status_code == 422


def test_abbreviate_requires_title(client):
    """POST /abbreviate requires title."""
    response = client.post("/abbreviate", json={})
    assert response.status_code == 422


def test_reabbreviate_requires_arxiv_id(client):
    """POST /papers/reabbreviate requires arxiv_id."""
    response = client.post("/papers/reabbreviate", json={})
    assert response.status_code == 422


def test_save_paper_requires_fields(client):
    """POST /papers/save requires arxiv_id, title, and authors."""
    response = client.post("/papers/save", json={})
    assert response.status_code == 422


def test_batch_ingest_requires_source(client):
    """POST /papers/batch-ingest requires directory or slack_channel."""
    response = client.post("/papers/batch-ingest", json={})
    if response.status_code == 503:
        if os.environ.get("REQUIRE_EXTERNAL_ENDPOINTS"):
            pytest.fail("LLM endpoint not available (REQUIRE_EXTERNAL_ENDPOINTS=1)")
        pytest.skip("LLM endpoint not available")
    assert response.status_code == 400


def test_prefetch_requires_fields(client):
    """POST /papers/prefetch requires arxiv_id and title."""
    response = client.post("/papers/prefetch", json={})
    assert response.status_code == 422


def test_repo_search_requires_fields(client):
    """POST /repos/search requires arxiv_id and title."""
    response = client.post("/repos/search", json={})
    assert response.status_code == 422


def test_references_fetch_requires_arxiv_id(client):
    """POST /references/fetch requires arxiv_id."""
    response = client.post("/references/fetch", json={})
    assert response.status_code == 422


def test_references_explain_requires_fields(client):
    """POST /references/explain requires reference_id, source_paper_title, cited_title."""
    response = client.post("/references/explain", json={})
    assert response.status_code == 422


def test_similar_requires_arxiv_id(client):
    """POST /papers/similar requires arxiv_id."""
    response = client.post("/papers/similar", json={})
    assert response.status_code == 422


def test_tree_node_requires_fields(client):
    """POST /tree/node requires node_id, name, and node_type."""
    response = client.post("/tree/node", json={})
    assert response.status_code == 422
