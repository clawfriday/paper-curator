# Functional Tests

Functional tests verify individual backend capabilities work correctly. These tests require external services (LLM endpoints, embedding endpoints, external APIs) but test one capability at a time.

## Prerequisites

- Backend running at `BACKEND_URL` (default: http://localhost:3100)
- LLM endpoint accessible
- Embedding endpoint accessible
- Sample PDFs in `tests/storage/downloads/`

## Running Tests

```bash
cd tests/functional
BACKEND_URL=http://localhost:3100 pytest -v -s
```

---

## Test Cases

### test_paper_download

Download the "Attention Is All You Need" paper (arxiv:1706.03762) into `tests/storage/downloads/`.

Inputs:
- arxiv_id: "1706.03762"
- output_dir: "tests/storage/downloads"

Success criteria:
- Response status 200
- PDF file exists at expected path
- PDF file size > 100KB (not corrupted/empty)
- PDF filename contains arxiv_id

---

### test_pdf_extraction

Extract text from a downloaded PDF and verify quality.

Inputs:
- pdf_path: "tests/storage/downloads/1706.03762.pdf"

Success criteria:
- Response status 200
- Extracted text length > 5000 characters
- Text contains expected keywords: "attention", "transformer", "encoder", "decoder"
- Number of pages > 0
- No error fields in response

---

### test_embedding

Verify chunk-wise and full-text embeddings are generated correctly.

Inputs:
- For abstract embedding: sample abstract text from "Attention Is All You Need"
- For fulltext: arxiv_id "1706.03762" with pdf_path

Success criteria for chunk embedding:
- Response status 200
- Embedding is a non-empty list of floats
- Embedding dimension matches expected (768 or 1536 depending on model)
- All values are valid floats (not NaN or Inf)

Success criteria for fulltext embedding:
- Index file created at storage/paperqa_index/{arxiv_id}.pkl
- Document embedding stored in database
- Embedding is mean-pooling of chunk embeddings (verify by recomputing)

---

### test_summarize

Generate a summary for a paper.

Inputs:
- pdf_path: "tests/storage/downloads/1706.03762.pdf"
- arxiv_id: "1706.03762"

Success criteria:
- Response status 200
- Summary text length > 200 characters
- Summary contains key concepts from the paper
- prompt_id field present in response
- No error fields in response

---

### test_paper_abbreviate

Generate an abbreviation for a paper title.

Inputs:
- title: "Attention Is All You Need"

Success criteria:
- Response status 200
- Abbreviation length between 2 and 15 characters
- Abbreviation is non-empty string
- Abbreviation contains alphanumeric characters (not just punctuation)

---

### test_paper_reabbreviate

Regenerate abbreviation for an existing paper and verify tree update.

Inputs:
- arxiv_id of a paper already in the database

Success criteria:
- Response status 200
- New abbreviation returned in response
- Tree node for this paper has updated name matching new abbreviation
- Database paper record has updated abbreviation field

---

### test_classify

Classify a paper into a category.

Inputs:
- title: "Attention Is All You Need"
- abstract: first paragraph of the paper's abstract
- existing_categories: ["Computer Vision", "NLP", "Reinforcement Learning", "Speech"]

Success criteria:
- Response status 200
- Category field present and non-empty
- Category is semantically reasonable for the paper (should be NLP-related)

---

### test_structured_analysis

Run structured Q&A analysis on an indexed paper.

Inputs:
- arxiv_id: "1706.03762" (must be indexed first)

Success criteria:
- Response status 200
- Response contains "queries" list with multiple Q&A pairs
- Each Q&A pair has "question" and "answer" fields
- Answers are non-empty and relevant to questions
- Use LLM to verify: ask "Do these Q&A pairs accurately describe the Transformer architecture paper?" - expect affirmative

---

### test_query_paper

Ask a specific question about a paper.

Inputs:
- arxiv_id: "1706.03762"
- question: "What is the computational complexity of self-attention?"

Success criteria:
- Response status 200
- Answer field present and length > 50 characters
- Answer mentions complexity notation (O(n²) or similar)
- Use LLM to verify: ask "Is this answer correct regarding attention complexity?" - expect affirmative

---

### test_fetch_github_repo

Fetch GitHub repositories for a paper with known repos.

Inputs:
- arxiv_id: "1706.03762"
- title: "Attention Is All You Need"

Success criteria:
- Response status 200 (or 503 if GitHub API unavailable - skip test)
- repos list contains at least one repository
- Each repo has: name, url, stars fields
- At least one repo URL contains "github.com"
- Popular paper should have repos with stars > 100

---

### test_fetch_reference

Fetch references for a paper.

Inputs:
- arxiv_id: "1706.03762"

Success criteria:
- Response status 200 (or 503 if Semantic Scholar unavailable - skip test)
- references list is non-empty
- Each reference has: title field
- References count > 10 (well-cited paper)

---

### test_fetch_similar

Fetch similar papers for a paper.

Inputs:
- arxiv_id: "1706.03762"

Success criteria:
- Response status 200 (or 503 if Semantic Scholar unavailable - skip test)
- similar_papers list is non-empty
- Each similar paper has: title, arxiv_id or paper_id
- Similar papers are semantically related (NLP/Transformer domain)

---

### test_topic_search

Search for papers matching a topic.

Inputs:
- topic: "self-attention mechanisms"
- limit: 10

Success criteria:
- Response status 200
- papers list returned (may be empty if no papers in DB match)
- topic_embedding field present with correct dimensions
- If papers returned, each has paper_id and similarity score
- Similarity scores are between 0 and 1

---

### test_tree_structure

Verify tree structure is valid.

Inputs:
- None (reads current tree state)

Success criteria:
- Response status 200
- Root node has node_id "root" and node_type "category"
- All nodes have required fields: node_id, node_type, name
- No duplicate node_ids
- All paper nodes have corresponding paper_id
- No cycles in tree structure (traverse without infinite loop)
