# Test Suite Documentation

## test_health.py

- `test_health`: Input `GET /health`, expect `{"status": "ok"}`; verifies backend is running and basic routing works.

## test_arxiv.py

- `test_arxiv_resolve_with_id`: Input `POST /arxiv/resolve` with `arxiv_id=1706.03762`; expect response includes `arxiv_id`, `title`, `authors`, `summary`, `pdf_url`, and title contains "Attention"; validates arXiv resolution via ID.
- `test_arxiv_resolve_no_identifier`: Input `POST /arxiv/resolve` with empty body; expect `400` with "Provide arxiv_id or url"; validates parameter validation.

## test_cached_data.py

- `test_cached_data_structured_summary`: Input creates a paper in DB, writes a structured summary, then calls `GET /papers/{arxiv_id}/cached-data`; expect response contains the same `structured_summary`; validates persistence and restore path for structured analysis.

## test_classify.py

- `test_classify_endpoint`: Input `POST /papers/classify`; expect response includes `message`, `papers_classified`, `clusters_created`; validates classification endpoint responds and returns expected keys (skips if external LLM/embedding endpoints are unavailable).

## test_tree.py

- `test_tree_endpoint`: Input `GET /tree`; expect response includes `name` and `children` list; validates tree endpoint returns frontend-compatible structure.

## test_delete_paper.py

- `test_delete_paper_endpoint`: Input creates a paper in DB then calls `DELETE /papers/{arxiv_id}`; expect deletion success (200/404) and follow-up `GET /papers/{arxiv_id}/cached-data` returns 404; validates delete endpoint and cascade cleanup.

## test_endpoints_validation.py

- `test_config_endpoint`: Input `GET /config`; expect UI config keys (`hover_debounce_ms`, `max_similar_papers`, `tree_auto_save_interval_ms`); validates config endpoint.
- `test_arxiv_download_requires_identifier`: Input `POST /arxiv/download` with empty body; expect 400 and "Provide arxiv_id or url"; validates identifier requirement.
- `test_pdf_extract_requires_path`: Input `POST /pdf/extract` with empty body; expect 422; validates pdf_path requirement.
- `test_summarize_structured_requires_pdf`: Input `POST /summarize/structured` with empty body; expect 422; validates pdf_path requirement.
- `test_embed_requires_text`: Input `POST /embed` and `/embed/abstract` with empty body; expect 422; validates text requirement.
- `test_embed_fulltext_requires_fields`: Input `POST /embed/fulltext` with empty body; expect 422; validates arxiv_id/pdf_path requirement.
- `test_qa_requires_question`: Input `POST /qa` with empty body; expect 422; validates question requirement.
- `test_qa_structured_requires_arxiv_id`: Input `POST /qa/structured` with empty body; expect 422; validates arxiv_id requirement.
- `test_summary_merge_requires_fields`: Input `POST /summary/merge` with empty body; expect 422; validates arxiv_id/selected_qa requirement.
- `test_summary_dedup_requires_arxiv_id`: Input `POST /summary/dedup` with empty body; expect 422; validates arxiv_id requirement.
- `test_classify_requires_fields`: Input `POST /classify` with empty body; expect 422; validates title/abstract requirement.
- `test_abbreviate_requires_title`: Input `POST /abbreviate` with empty body; expect 422; validates title requirement.
- `test_reabbreviate_requires_arxiv_id`: Input `POST /papers/reabbreviate` with empty body; expect 422; validates arxiv_id requirement.
- `test_save_paper_requires_fields`: Input `POST /papers/save` with empty body; expect 422; validates arxiv_id/title/authors requirement.
- `test_batch_ingest_requires_source`: Input `POST /papers/batch-ingest` with empty body; expect 400 (or skip if LLM endpoint unavailable); validates directory/slack_channel requirement.
- `test_prefetch_requires_fields`: Input `POST /papers/prefetch` with empty body; expect 422; validates arxiv_id/title requirement.
- `test_repo_search_requires_fields`: Input `POST /repos/search` with empty body; expect 422; validates arxiv_id/title requirement.
- `test_references_fetch_requires_arxiv_id`: Input `POST /references/fetch` with empty body; expect 422; validates arxiv_id requirement.
- `test_references_explain_requires_fields`: Input `POST /references/explain` with empty body; expect 422; validates reference_id/source_paper_title/cited_title requirement.
- `test_similar_requires_arxiv_id`: Input `POST /papers/similar` with empty body; expect 422; validates arxiv_id requirement.
- `test_tree_node_requires_fields`: Input `POST /tree/node` with empty body; expect 422; validates node_id/name/node_type requirement.
