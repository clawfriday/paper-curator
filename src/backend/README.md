# Backend API Documentation

This document describes the working logic of each API endpoint in the paper-curator backend.

## Technology Stack

- **FastAPI**: Web framework for API endpoints
- **PaperQA2**: PDF parsing, document indexing, and RAG-based Q&A
- **LiteLLM**: LLM abstraction layer (used internally by PaperQA2)
- **AsyncOpenAI**: Direct LLM calls for classification, abbreviation, etc.
- **httpx**: Async HTTP client for external API calls

---

## Health & Configuration

### `GET /health`
Simple health check endpoint.

**Returns**: `{"status": "ok"}`

---

### `GET /config`
Returns UI configuration loaded from `config/config.yaml`.

**Returns**: UI settings (tree layout, limits, etc.)

---

## arXiv Operations

### `POST /arxiv/resolve`
Resolve arXiv metadata from ID or URL.

**Steps**:
1. Parse arXiv identifier from `arxiv_id` or `url`
2. Query arXiv API using `arxiv` Python library
3. Return paper metadata (title, authors, abstract, categories)

**Uses**: arXiv API

---

### `POST /arxiv/download`
Download PDF and LaTeX source from arXiv.

**Steps**:
1. Parse arXiv identifier
2. Query arXiv API for paper metadata
3. Download PDF to `storage/downloads/{arxiv_id}.{title}.pdf`
4. Download LaTeX source to `storage/downloads/{arxiv_id}.{title}.tar.gz`

**Uses**: arXiv API

---

## PDF Processing

### `POST /pdf/extract`
Extract text content from a PDF file.

**Steps**:
1. Load PDF from provided path
2. Use PaperQA2's native PDF parser (`read_doc`)
3. Return extracted text content

**Uses**: PaperQA2 (PDF parsing)

---

## Summarization

### `POST /summarize`
Generate a single-query summary of a paper. Also indexes PDF for faster subsequent QA.

**Steps**:
1. Load prompt template from `prompts/prompts.json` (paper_summary_v2)
2. Get LLM and embedding model endpoints from config
3. If `arxiv_id` provided, check for cached PaperQA2 index
4. If no cache: parse PDF, chunk text, generate embeddings, create index
5. Run PaperQA2 query with summary prompt
6. Persist index to `storage/paperqa_index/{arxiv_id}.pkl` for reuse
7. Clean citation markers from response

**Uses**: PaperQA2 (indexing + RAG query), LLM via LiteLLM

---

### `POST /summarize/structured`
Generate a multi-component structured analysis of a paper.

**Steps**:
1. Get LLM and embedding model endpoints
2. Reset LiteLLM callbacks to prevent accumulation
3. **Step 1**: Extract key components using PaperQA2 query with `extract_components` prompt
4. Parse JSON array of components from LLM response
5. **Step 2**: For each component, run 4 sequential PaperQA2 queries:
   - `component_steps`: Logical chain of steps / pseudo-code
   - `component_benefits`: Main area it benefits
   - `component_rationale`: Rationale behind the benefit
   - `component_results`: Quantifiable results
6. Reset callbacks between each query to prevent LiteLLM accumulation
7. Return structured sections

**Uses**: PaperQA2 (multiple RAG queries), LLM via LiteLLM

---

## Embedding

### `POST /embed/abstract`
Generate embedding vector for abstract text (for pgvector similarity search).

**Steps**:
1. Get embedding model endpoint from config
2. Call embedding API with text
3. Return embedding vector

**Uses**: AsyncOpenAI (embedding endpoint)

---

### `POST /embed/fulltext`
Index a full PDF for PaperQA2 queries. Creates persistent index for reuse.

**Steps**:
1. Get LLM and embedding model endpoints
2. Parse PDF using PaperQA2
3. Chunk text and generate embeddings
4. Persist index to `storage/paperqa_index/{arxiv_id}.pkl`

**Uses**: PaperQA2 (indexing)

---

### `POST /embed` (deprecated)
Backwards-compatible alias for `/embed/abstract`.

---

## Question Answering

### `POST /qa`
Answer a question about a paper using RAG.

**Steps**:
1. Get LLM and embedding model endpoints
2. If `arxiv_id` provided, try to load cached PaperQA2 index
3. If no cache and `pdf_path` provided, index the PDF
4. Run PaperQA2 query with the question
5. Clean citation markers from response

**Uses**: PaperQA2 (RAG query), LLM via LiteLLM

---

### `POST /qa/structured`
Run detailed structured analysis on an already-indexed paper. Fully parallelized.

**Steps**:
1. Verify cached index exists for the paper
2. Get LLM and embedding model endpoints
3. **Step 1**: Run component extraction query (sequential)
4. Parse JSON array of components (max 5)
5. **Step 2**: Build list of all aspect queries (5 components × 4 aspects = 20 queries)
6. Run ALL 20 queries in parallel using `asyncio.gather`
7. Reassemble results into structured sections

**Uses**: PaperQA2 (parallel RAG queries), LLM via LiteLLM

**Note**: ~7x faster than sequential execution due to full parallelization.

---

## Classification & Abbreviation

### `POST /classify`
Classify a paper into a category using LLM.

**Steps**:
1. Get LLM endpoint from config
2. Load `classify` prompt template
3. Call LLM with title and abstract
4. Return predicted category

**Uses**: AsyncOpenAI (direct LLM call)

---

### `POST /abbreviate`
Generate a short abbreviation for a paper title.

**Steps**:
1. Get LLM endpoint from config
2. Load `abbreviate_v2` prompt template
3. Call LLM with title
4. Return abbreviation (max 15 chars, preserves version numbers)

**Uses**: AsyncOpenAI (direct LLM call)

---

### `POST /papers/reabbreviate`
Re-generate abbreviation for an existing paper.

**Steps**:
1. Get paper from database by arxiv_id
2. Call `/abbreviate` logic with paper title
3. Update tree node with new abbreviation

**Uses**: AsyncOpenAI (direct LLM call)

---

### `POST /papers/reabbreviate-all`
Re-generate abbreviations for ALL papers in parallel.

**Steps**:
1. Get all papers from database
2. Create abbreviation tasks for each paper
3. Run all tasks in parallel using `asyncio.gather`
4. Update all tree nodes

**Uses**: AsyncOpenAI (parallel LLM calls)

---

## Paper Management

### `POST /papers/save`
Save a paper to the database and add it to the tree.

**Steps**:
1. Create paper record in PostgreSQL database
2. Generate embedding for abstract (for similarity search)
3. Store embedding in pgvector column
4. Add node to tree structure

---

### `POST /papers/batch-ingest`
Batch ingest PDFs from a local directory.

**Steps**:
1. Map host path to Docker container path (`/Users/xxx` → `/host_home/xxx`)
2. Find all PDF files in directory
3. For each PDF:
   a. Copy to `storage/downloads`
   b. Extract text using PaperQA2
   c. Generate title from first line or filename
   d. Run classify, abbreviate, and summarize in parallel
   e. Reset LiteLLM callbacks between papers
   f. Save to database and add to tree

**Uses**: PaperQA2 (extraction), AsyncOpenAI (classify/abbreviate), LiteLLM (summarize)

---

## Tree Operations

### `GET /tree`
Get the full tree structure.

**Returns**: Nested tree with root node "Research Papers" and all paper nodes.

---

### `POST /tree/node`
Add a node to the tree.

**Parameters**: `node_id`, `name`, `parent_id`, `attributes`

---

### `DELETE /tree/node/{node_id}`
Delete a node from the tree.

---

## External Data Fetching

### `POST /papers/prefetch`
Prefetch repos, references, and similar papers in parallel.

**Steps**:
1. Create 3 async tasks:
   - Search GitHub repos
   - Fetch Semantic Scholar references
   - Find similar papers
2. Run all 3 in parallel using `asyncio.gather`
3. Return combined results

**Uses**: GitHub API, Semantic Scholar API

---

### `POST /repos/search`
Search for GitHub repositories associated with a paper.

**Steps**:
1. Check cache for existing results
2. Search Papers With Code API for paper's GitHub links
3. If no results, search GitHub API directly
4. Cache results and return

**Uses**: Papers With Code API, GitHub Search API

---

### `POST /references/fetch`
Fetch references for a paper.

**Steps**:
1. Get paper from database
2. Search Semantic Scholar API for paper
3. Fetch reference list
4. Return reference metadata

**Uses**: Semantic Scholar API

---

### `POST /references/explain`
Generate an explanation for why a reference is relevant.

**Steps**:
1. Check cache for existing explanation
2. Load `reference_explanation` prompt template
3. Call LLM with paper context and reference info
4. Cache and return explanation

**Uses**: AsyncOpenAI (direct LLM call)

---

### `POST /papers/similar`
Find similar papers using Semantic Scholar Recommendations API.

**Steps**:
1. Get paper from database
2. Search Semantic Scholar for paper ID
3. Call Recommendations API with paper ID
4. Return list of similar papers

**Uses**: Semantic Scholar Recommendations API

---

## Connection Pooling

The backend uses shared `httpx.AsyncClient` pools for external APIs:
- Semantic Scholar
- GitHub
- Papers With Code

This eliminates TLS handshake overhead on repeated calls (~17% speedup).

---

## LiteLLM Callback Management

PaperQA2 uses LiteLLM internally, which accumulates callbacks with each LLM call. The `_reset_litellm_callbacks()` function clears all callback lists to prevent the MAX_CALLBACKS (30) limit from being reached during batch operations.

---

## PaperQA2 Settings

Key settings in `_build_paperqa_settings()`:
- `use_json=False`: Disables strict JSON parsing to avoid errors with non-OpenAI models
- `chunk_chars`: Characters per chunk (default 3000)
- `chunk_overlap`: Overlap between chunks (default 100)
- `evidence_k`: Number of evidence passages to retrieve (default 10)
