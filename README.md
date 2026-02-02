# Paper Curator

AI-powered paper reading and curation pipeline with interactive GUI.

## Quick Start

### Prerequisites

- **Python 3.11+** (for backend)
- **Node.js 18+** and **npm** (for frontend)
- **Docker** and **Docker Compose** (for running services)
- **PostgreSQL** (via Docker)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd paper-curator
   ```

2. **Install all dependencies** (Python + npm)
   ```bash
   make install
   ```
   This will:
   - Create a Python virtual environment (`.venv`)
   - Install Python dependencies from `pyproject.toml`
   - Install npm dependencies from `src/frontend/package.json` and `package-lock.json`

3. **Start services**
   ```bash
   make run
   ```
   This starts all services (database, backend, frontend) via Docker Compose.

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Database: localhost:5432

### Development

- **Run tests**: `make test`
- **Run backend locally**: `make run-local` (requires `.venv` activated)
- **Build Docker images**: `make docker-build`
- **Stop services**: `make docker-stop`
- **Clean up**: `make clean`

---

# Project Scope

This is a pipeline with interactive graphics to automate the workflow of AI paper reading and curation. The main graph is a tree which classifies the papers into different categories/sub-categories based on their topic. This tree diagram needs to be interactive, such that a new paper can be added into it, and the tree diagram will be update afterwards. It will also allow user to click on each paper to perform additional operations. It will contain the following requirements:

1. ingest_paper
- take an user input of URL from arxiv (alternatively it might be the pdf itself, then you need to identify its source URL from internet) 
- download both the pdf, Latex code (if available) from its arxiv link into structured database

2. parse_pdf
- read the paper from pdf
- compute its embeddings
- provide a concise summary in markdown format include the aspects: 
    - what it does concretely, it needs to be a reasonable logical chain of thoughts, in which the rationale of the previous step justifies the next step, using plain language as much as possible, avoid confusing technical jargons (unless it's specifically named in the paper itself), avoid introducing figurative speech 
    - which area it mainly benefits (model performance / training efficiency / inference throughput / etc.). 
    - the rationale behind this benefit 
    - no duplication in the above content 
    - quantifiable results in the main area it is trying to improve 
    - if the paper contains multiple unique concepts (like DeepSeek-V3 technical report contains MTP, MLA, fine-grain FP8 etc), which are uncorrelated, please futher break the paper into multiple leave nodes (e.g. DeepSeek-v3-MTP, DeepSeek-v3-MLA, etc.) 

3. classify_paper
- adding each new paper as new node to the tree of classification
- using the embedding of this paper, classify the papers by their respective main area of contribution, e.g. primary level of classification can be any of the AI areas (e.g. dataset, evaluation, model architecture, inference, application, vision, speech, linguistic, RL etc). 
- if each category is getting crowded (>10), when add the new paper node, we need to review the classification to determine how shall we further branch out the current category into new sub-classes (using LLM to do this sub class creation and sub-classification), and then migrate the leave nodes under it into the new sub-classes 
- udpate the tree with the latest results of classification

4. git_repo
for each paper, the 1st option (from right-click dropdown menu) will be to obtain its open-source repo (if available), if no official one is available, identify if open-source replication is available as well. 

5. explain_references
for each paper, the 2nd option required is to "explain on key references". Once clicked, it will give me a full list of references in this paper, I can then:
- hover my mouse over each reference, it will trigger a LLM workflow to provide a concise description of this reference, and how is the reference related/contributing to the current paper
- click on the reference, so that it will add the reference into this diagram as a new paper, following all steps above

6. find_similar
for each paper, the 3rd option required is to "find similar papers". Once clicked, it will search in arxiv or google for the most similar 2-5 papers. Again, each will have a concise description of what this reference is about, and how is it similar to the paper 

7. all above info will be persisted in a locally structured database. please help me evaluate if such plan is feasible, and what are the key steps I need to take in n8n to implement such

8. move the prompts
- move the prompts from 'prompts' to 'src/backend/prompts', 
- change it from yaml to json. 
- move all existing prompts that are hardcoded into the python script into the json
- load it only when we need to use the prompt

# Resource

## vllm endpoint that you may need to use
LLM: OPENAI_API_BASE="http://localhost:8001"
VLM: OPENAI_API_BASE2="http://localhost:8002"
Embed: OPENAI_API_BASE3="http://localhost:8004"

## OSS tools

- arxiv.py: Reliable library for metadata + PDF download, can be used in requirement #1

- GROBID: PDF parsing & structured citation extraction, can be used in requirement #2

- PaperQA2 (paper-qa): Summarization + concept extraction + multi-turn QA, Scientifically tuned RAG wrapper, can be used in requirement #2 and #4

- PostgreSQL + pgvector: Database + embeddings + similarity search, can be used in all requirements

- react-d3-tree: Local front-end visualization and curation UI, can be used in all requirements

## Implementation Requirement
modularize the implementation, let's start bottomeup
- first install each of the OSS tools, test if the are working
- if so, then package each as a local MCP server, and then an agent skill, add such agent skill in cursor
- then let the cursor agent run the "Scope" as defined here, with such skills
- standardize runtime in the **single GROBID-based container** (no local uv/venv)

---

# Implementation Plan
This follows your ordering: **(1) package OSS as FastAPI services**, **(2) build internal blocks (UI + missing logic) and only then DB**, **(3) wire via LangGraph**, **(4) pause and reassess**.

### Milestone 1 — package OSS tools as local FastAPI services (with scripts)
- **Goal**: each OSS tool is callable locally via HTTP, with runnable scripts and known failure modes.
- Deliverables (single service):
  - `paper-curator-service`: arXiv resolve + download + GROBID extract + summarize + embed + optional QA
- **Local GROBID setup** (Docker):
  - Pull: `docker pull lfoppiano/grobid:0.8.0`
  - Build (unified container): `bash scripts/docker_mod.sh`
  - Run (unified container): `bash scripts/docker_run.sh`
  - Health: `GET ${service_base_url}/api/isalive`
- **Endpoint list**
  - `paper-curator-service`
    - `GET /health`
    - `POST /arxiv/resolve`
    - `POST /arxiv/download`
    - `POST /pdf/extract` (GROBID-backed)
    - `POST /summarize`
    - `POST /embed`
    - `POST /qa` (optional)
- **Scripts (replace unit tests)**:
  - One script per endpoint under `scripts/` (see `scripts/README.md`)
- **Exit criteria**: services run locally, `/health` passes, scripts succeed end-to-end for 2–3 representative papers.


### PaperQA2 findings (from paper-qa/README + code)
- **What we used naively vs built-in PaperQA2**
  - We manually chunk text and feed `Docs.aadd_texts`; PaperQA2 already provides parsing + chunking (`Docs.aadd` / `Docs.aadd_file`) with configurable chunk size/overlap, metadata validation, and PDF parsing hooks.
  - We run a simple summarize/QA path; PaperQA2’s `Docs.aget_evidence` + `Docs.aquery` implements retrieval, contextual summarization (summary LLM), and LLM re-ranking before answer generation.
  - We skip metadata inference; PaperQA2 can infer/attach citations, title/DOI/authors, and uses metadata in embeddings and ranking.
  - We bypass agentic search and evidence tooling; PaperQA2 provides agentic workflows (search → gather evidence → answer) and a “fake” deterministic path for lower token usage.

- **Capabilities we can still leverage for other scope sections**
  - Paper search + metadata aggregation across providers (Crossref, Semantic Scholar, OpenAlex, Unpaywall, retractions, journal quality).
  - Full-text indexing + reuse (local index, cached answers, `search` over prior answers/documents).
  - Hybrid/sparse+dense embeddings, configurable `evidence_k`, evidence relevance cutoff, and MMR settings.
  - Multimodal parsing/enrichment for figures/tables (media-aware contextual summaries).
  - External vector stores / caching hooks, plus settings presets and CLI profiles for reproducible runs.
  - Code/HTML/Office document ingestion for repository or artifact QA.

### Milestone 2 — internal building blocks (no DB yet)
- **Interactive UI** (React + `react-d3-tree`)
  - Render a local taxonomy state (file/in-memory)
  - Add “ingest paper” input (arXiv URL) and show status/progress
  - Node details panel: summary markdown + references list
- **Non-OSS requirements**
  - Taxonomy maintenance logic (crowding split + migration)
  - Hover caching + debouncing policy for reference explanations
  - Reference resolution + dedupe policy (canonical IDs, fuzzy title match)

### Milestone 3 — LangGraph orchestration
- Build LangGraph graphs for Flows A–F:
  - `ingest_paper_graph`, `parse_and_summarize_graph`, `classify_and_update_tree_graph`
  - UI action graphs: `repo_lookup_graph`, `explain_reference_graph`, `find_similar_graph`
- **Exit criteria**: UI triggers a LangGraph run and receives structured results; failures are surfaced with clear errors + partial outputs where possible.

### Milestone 4 — database
- Introduce PostgreSQL + `pgvector` for durability + search once the behaviors stabilize:
  - Persist papers/artifacts/embeddings
  - Persist taxonomy nodes/edges + rationales
  - Cache hover explanations + similarity results

### Mileston 5 - refactor
- using split Docker for frontend and backend, docker-bridge for comm via compose.yml
- using venv for dev
- use Makefile to simplify the dev endpoint


### Mileston 6 - UI improvement (Complete)

**Phase 1: Details Panel Modernization**
- ✅ Replaced custom tabs with Shadcn/ui Tabs component (`@radix-ui/react-tabs`)
- ✅ Replaced collapsible sections with Shadcn/ui Accordion component (`@radix-ui/react-accordion`)
- ✅ Replaced custom cards with Shadcn/ui Card component (`@radix-ui/react-slot`)
- ✅ Replaced custom tooltips with Shadcn/ui Tooltip component (`@radix-ui/react-tooltip`)
- ✅ Implemented consistent spacing system using Tailwind CSS (`tailwindcss`, `tailwindcss-animate`)
- ✅ Added utility functions (`class-variance-authority`, `clsx`, `tailwind-merge`)
- **Dependencies:** `tailwindcss`, `postcss`, `autoprefixer`, `tailwindcss-animate`, `class-variance-authority`, `clsx`, `tailwind-merge`, `@radix-ui/react-tabs`, `@radix-ui/react-accordion`, `@radix-ui/react-tooltip`, `@radix-ui/react-slot`, `lucide-react`

**Phase 2: Tree Diagram Enhancement**
- ✅ Replaced react-d3-tree with React Flow (`reactflow`, `@xyflow/react`)
- ✅ Implemented custom PaperNodeComponent with React Flow Handle components
- ✅ Configured hierarchical tree layout with proper node positioning
- ✅ Added root node ("AI Papers") with connections to top-level categories
- ✅ Implemented edge rendering with smoothstep connections
- ✅ Added React Flow controls (zoom, pan, minimap)
- **Dependencies:** `reactflow`, `@xyflow/react`

**Phase 3: Navigation & Interaction**
- ✅ Added global search bar in header with paper title/author search
- ✅ Implemented real-time search with dropdown results (up to 10 matches)
- ✅ Added search icon using Lucide React (`lucide-react`)
- **Dependencies:** `lucide-react` (Search icon)

**Phase 4: Responsive Design & Polish**
- ✅ Implemented responsive breakpoints (mobile: <768px, tablet: 768-1024px, desktop: >1024px)
- ✅ Added panel resizing/dragging capability between tree and details panels
- ✅ Added fullscreen mode toggle for tree or details panel
- ✅ Moved logs to expandable debug panel (collapsed by default) using Shadcn/ui Accordion
- ✅ Added loading skeletons for async content (repos, references, similar papers, queries)
- ✅ Implemented hover previews for paper nodes showing title only
- **Dependencies:** `tailwindcss` (responsive utilities), `@radix-ui/react-accordion` (debug panel)

**Phase 5: Layout**
- change the root node 'AI papers' and parent nodes (different classifications of all levels) placement to be horizontal
- however, keep the leave nodes (papers) to vertical placement 

**Phase 6: Simplify the panes**
- remove the top-right Pane (Ingest Pane), and move its functions to a new tab 'explorer' in the current bottom-right Pane
- unify the `Ingest` and `Batch Ingest`, it should only have one input textbox, one button, and one unified endpoint. The input box will trigger different actions based on the detected type of input text:
  - if it's a URL, or local filepath to a single paper, or a Arxiv paper ID, it will just do a single injestion
  - if it's a local folder, which contains multiple pdf papers, it will trigger a batch ingestion
  - if it's a slack channel, it will trigger batch ingestions (a feature to be implemented later)
- the "rebalance" should just be a single small button in the "explorer" tab, instead of taking up the entire row
- instead of using dropdown menus when rightclick on each node, let's move each dropdown option to the respective tabs as a new tab

### Mileston 7 - Speedup (Complete)

#### Step Timings with Optimizations

| Step | Before | After | Speedup | Implementation |
|------|--------|-------|---------|----------------|
| Extract + Classify + Abbreviate | 2.84s (sequential) | 1.98s (parallel) | **30%** | Frontend Promise.all |
| QA Query | 39s | 24s | **38%** | Persist PaperQA2 index |
| Bulk Re-abbreviate (6 papers) | 1.89s | 1.17s | **38%** | asyncio.gather |
| Prefetch (repos+refs+similar) | 1.24s | 0.64s | **48%** | asyncio.gather |
| External API requests | 46ms first | 38ms reuse | **17%** | Connection pooling |
| Structured QA (5 components) | ~10min (sequential) | ~1.5min (parallel) | **~7x** | asyncio.gather for all aspect queries |

#### Implemented Optimizations

1. **Parallel ingest steps**: Extract, Classify, and Abbreviate now run concurrently via Promise.all. All three are independent - extraction uses PDF, while classify/abbreviate use title/abstract.

2. **Persist PaperQA2 index**: Summarize now saves the indexed Docs object (chunks + embeddings) to `storage/paperqa_index/`. QA queries load the cached index instead of re-parsing the PDF.

3. **Async bulk operations**: Added `/papers/reabbreviate-all` endpoint using asyncio.gather for concurrent LLM calls. vLLM handles batching internally.

4. **Parallel prefetch**: Added `/papers/prefetch` endpoint that fetches repos, references, and similar papers concurrently. Frontend triggers this in background after ingest.

5. **Connection pooling**: Shared httpx.AsyncClient pool for Semantic Scholar, GitHub, and Papers With Code. Eliminates TLS handshake overhead on repeated calls.

6. **Parallel structured QA**: The `/qa/structured` endpoint now runs all aspect queries in parallel using `asyncio.gather`. Component extraction runs first (sequential), then all 20 aspect queries (5 components × 4 aspects) run concurrently, reducing total time from ~10min to ~1.5min.

#### New Endpoints
- `POST /embed/abstract` - Embed text for pgvector similarity
- `POST /embed/fulltext` - Index PDF for PaperQA2 (persisted)
- `POST /papers/reabbreviate-all` - Bulk re-abbreviate all papers
- `POST /papers/prefetch` - Prefetch repos/refs/similar in parallel

### Milestone 8 - bugfixes
- [x] meaningless "suma2025deepseekr1incentivizingreasoning" token from model response
- [x] the prompt on "abbreviation" should be less restrictive. Currently the technical report for Deepseek v3.2 is summarized as Deepseek v3, which conflicts with actual Deepseek v3 technical report
- [x] to breakdown the paper summary query into multile queries, each query should only focus on on one question. **Implemented via `/qa/structured` endpoint with full parallelization (asyncio.gather for all 20 aspect queries)**
- [x] the summary under 'Details' should be displayed in sections, each section contains the answer to one of the queries earlier. Each section should be collapsible
- [x] currently, the auto-reclassification wasn't triggered, even when the parent node (e.g. "Natural Language Process") is getting crowed. **Fixed**: Auto-reclassification now triggers when a category exceeds `category_threshold` (default 10) in config. Classification uses LLM with dynamic category generation - it can reuse existing categories or create new ones based on paper content.
- [x] the outcome of the following should be persisted into the database, and reloaded into GUI whenever the server restarts, including the following:
  - details: stored in `papers.summary` column
  - repos: stored in `repo_cache` table
  - refs: stored in `paper_references` table  
  - similar: stored in `similar_papers_cache` table
  - query: **NEW** stored in `paper_queries` table, loaded via `/papers/{arxiv_id}/cached-data` endpoint
- [x] Duplicate detection: after resolving arXiv ID, check if paper already exists in database; if so, skip ingestion with "already exists" message
- [x] Remove node: right-click menu on paper nodes includes "Remove Paper" option (red, with confirmation dialog); deletes paper and all associated data via `DELETE /papers/{arxiv_id}` endpoint
- [x] LaTeX formula rendering: added react-katex library to render LaTeX formulas in Details and Query panels; formulas in format \(...\) (inline) and \[...\] (block) are now properly rendered as mathematical notation


### Milestone 9 - add QA to summary
- [x] Query selection: each historical query in the Query tab has a checkbox; clicking the row or checkbox toggles selection
- [x] "Add to Details" button: appears below query history, disabled when nothing selected; merges selected Q&A into summary using LLM (`/summary/merge` endpoint with `merge_qa_to_summary` prompt)
- [x] "Dedup" button: in Details panel action buttons; removes duplicated content from summary using LLM (`/summary/dedup` endpoint with `dedup_summary` prompt)ed 

### Mileston 10 - Slack integration
- allow user to specify a slack channel in the 'batch ingest' textbox
- it will prompt user for access credential to access the slack channel each time. But this credential should not be persisted
- it will read all historical paper shared into the channel, and ingest them all

### Milestone 10 - classification logic
- use document-level embedding, instead of abstract-only LLM call
- remove the 'reclassification' backend endpoint, leave with only the unified 'classification' endpoint
- remove the pre-defined category, use only dynamic definition of category name
- the classification algorithm goes in two main stages:
- Stage 1: tree building
  - obtain document-level embedding from each full pdf (it not already available)
  - hierarchical clustering based on document-level embedding
    - run clustering algorithm based on the document-level embedding of each pdf
    - each cluster will be assigned a cluster id, and built as a node in the tree diagram
    - whenever a cluster has > 3 ("Branching Factor" predefined in config) children nodes, branch it into sub-clusters
    - apply the branching to each cluster/ recursively, until a tree is built
    - each leaf node in the tree is a paper
- Stage 2: naming the nodes
  - a summary of each pdf should be available (from earlier steps during ingestion)
  - from bottom level upwards, the lowest level (level i) parent node will take the summary of all leave node (level i+1), to contrast with the summary of all other leave nodes under the same grandparent node (level i-1), and come up with a proper node-name for itself.
  - once done, the next higher level parent node (level i-1) will take the node-name (instead of summary) of all children node (level i) under it, to contrast with the other node-names under the same grandparent node (level i-2), and come up with proper node-name for itself
  - it will run this iteratively upward to the top level

## Implementation Plan: Milestone 10 - Embedding-Based Classification

### Overview
Replace LLM-based classification with embedding-based hierarchical clustering. Use mean pooling of PaperQA2 chunk embeddings for document-level representation.

### Phase 1: Foundation - Embedding Extraction & Infrastructure
1. **Add dependencies** (`src/backend/requirements.txt`):
   - `scikit-learn` (for divisive clustering)
   - `scipy` (for distance metrics)

2. **Add configuration** (`config/paperqa.yaml`):
   ```yaml
   classification:
     branching_factor: 3  # Max children before branching
     clustering_method: "divisive"  # Top-down divisive clustering
     rebuild_on_ingest: true  # Auto-rebuild tree on new paper
   ```

3. **Add mean pooling function** (`src/backend/app.py`):
   - Extract chunk embeddings from PaperQA2 `Docs` object (access via `docs.docs[].texts[].embedding`)
   - Compute mean pooling: `doc_embedding = mean([chunk.embedding for chunk in all_chunks])`
   - Store in `papers.embedding` column using `db.update_paper_embedding()`

4. **Integrate into ingestion workflow**:
   - After PaperQA2 indexing in `_index_pdf_for_paperqa_async()`, extract and save document embedding
   - Update both single ingest (`/papers/ingest`) and batch ingest (`/papers/batch-ingest`)
   - Reuse existing embedding if already computed (check DB before computing)

### Phase 2: Core Algorithm - Clustering & Naming
**New files**: `src/backend/clustering.py`, `src/backend/naming.py`

1. **Divisive clustering & tree building** (`clustering.py`):
   - Load all papers with embeddings from DB
   - Use top-down divisive clustering (recursive k-means or similar)
   - Build tree structure recursively:
     - Start with all papers as root cluster
     - If cluster size > `branching_factor`, split into sub-clusters
     - Recursively apply to each sub-cluster
     - Create category nodes for each cluster
     - Link papers to their cluster nodes
   - Tree rebuild function: Clear all existing category nodes, run clustering, rebuild tree in `tree_state` table

2. **Contrastive naming** (`naming.py`):
   - Add prompt template `node_naming` in `prompts.json`: Takes children summaries/names and sibling summaries/names, outputs distinguishing category name
   - Bottom-up naming function:
     - Traverse tree level by level (bottom-up)
     - For each parent node: collect summaries/names of children and siblings, call LLM with contrastive prompt, update node name

### Phase 3: Integration - Endpoints, Frontend & Migration
1. **New unified endpoint** (`src/backend/app.py`):
   - `POST /papers/classify`: Optional `arxiv_id` input, runs clustering + tree building + naming, returns tree structure
   - Update ingestion flow: After ingesting, if `rebuild_on_ingest: true`, trigger tree rebuild
   - Remove old classification: Remove `classify` prompt usage, remove `/categories/rebalance` endpoint, remove auto-reclassification logic

2. **Frontend integration** (`src/frontend/src/app/page.tsx`):
   - Add "Re-classify" button in Explorer/Ingest tab (calls `POST /papers/classify`, shows progress, refreshes tree)
   - Remove "Rebalance" button if exists

3. **Migration** (`migrations/migrate_to_embedding_classification.py`):
   - For all existing papers: compute document embeddings (mean pooling from PaperQA2 index if available)
   - Clear all category nodes, rebuild tree from scratch, name all nodes

### File Changes Summary

**New files**:
- `src/backend/clustering.py` - Clustering and tree building
- `src/backend/naming.py` - Contrastive naming logic
- `migrations/migrate_to_embedding_classification.py` - Migration script

**Modified files**:
- `src/backend/app.py` - Add embedding extraction, new endpoint, remove old classification
- `src/backend/db.py` - Helper functions for tree operations
- `src/backend/prompts/prompts.json` - Add `node_naming`, remove `classify`
- `src/backend/requirements.txt` - Add scikit-learn, scipy
- `config/paperqa.yaml` - Add clustering config
- `src/frontend/src/app/page.tsx` - Add "Re-classify" button, remove "Rebalance"

## Phase 4: Rewrite clustering logic

### Problem Statement

The current `clustering.py` has several issues:
1. **Node ID collision**: Content-based ID generation (`_generate_node_id`) can produce the same ID for different tree positions when k-means degenerates to a single cluster
2. **Dual ID systems**: Both `node_` and `cluster_` prefixes are used, causing confusion and mismatch with `node_names`
3. **Two-phase tree building**: Raw tree is built first, then wrapped for frontend, introducing complexity and potential for cycles during wrapping
4. **Hand-written silhouette score**: Custom implementation instead of using sklearn's optimized version
5. **No embedding normalization**: K-means uses Euclidean distance, but document embeddings should use cosine similarity

### Modification Plan

#### 1. Fix empty cluster handling
- Add `len(clusters) <= 1` check after k-means to catch degenerate cases
- Use `BisectingKMeans` as fallback when standard KMeans produces empty clusters
- This prevents the ID collision scenario entirely

#### 2. Unify ID system - use `node_` prefix only
- Remove `_generate_cluster_id()` function
- Use `node_` prefixed IDs throughout (tree building, database, frontend)
- Ensures `node_names` mapping works correctly

#### 3. Build directly in frontend format
- Eliminate separate wrapping step (`_wrap_tree_for_db`)
- `_split_node()` returns nested dict directly: `{name, node_id, node_type, children}`
- Remove single-child intermediates inline during recursion
- Result is directly saveable to DB and renderable by frontend

#### 4. Use sklearn's silhouette_score
- Replace hand-written `_compute_cosine_silhouette_score()` with:
  ```python
  from sklearn.metrics import silhouette_score
  score = silhouette_score(embeddings, labels, metric='cosine')
  ```

#### 5. L2-normalize embeddings before clustering
- Normalize embeddings so k-means effectively uses cosine similarity:
  ```python
  from sklearn.preprocessing import normalize
  embeddings_normalized = normalize(embeddings, norm='l2')
  ```

#### 6. Code cleanup
- Remove silent error handling (log warnings instead)
- Remove obsolete functions and comments

### File Changes

**Modified files**:
- `src/backend/clustering.py` - Complete rewrite of tree building logic
- `src/backend/naming.py` - Updated documentation, cleaned up imports
- `src/backend/db.py` - Renamed `find_paper_cluster_id` to `find_paper_node_id`, updated terminology
- `src/backend/app.py` - Updated to use new db function names
- `tests/test_clustering.py` - Added tests for circular references and unique node IDs

### Naming Process (`naming.py`)

The naming module works with the new tree structure to provide LLM-generated names for category nodes.

**Process (bottom-up, level-by-level)**:
1. Get tree structure from database (frontend format with `node_id`, `node_type`, `children`)
2. Organize category nodes by depth level (deepest first)
3. For each level, process all nodes in parallel:
   - Gather children content:
     - For leaf level: paper summaries from database
     - For higher levels: already-named category names
   - Gather sibling content (other categories with same parent)
   - Call LLM with contrastive prompt to generate distinguishing name
   - Update name in database and in-memory tree
4. Proceed to next level (higher up)

**Key features**:
- Uses `node_id` (node_xxx format) consistently
- Updates both database and in-memory tree to keep them in sync
- Parallel naming within each level for speed
- Sequential level processing to ensure children are named before parents
- Fallback naming if LLM fails after retries

### Test Coverage

The `tests/test_clustering.py` now includes:
1. All papers have embeddings
2. Tree builds successfully
3. All papers present in tree
4. Branching factor constraint respected
5. No empty nodes (category nodes have children, paper nodes have paper_id)
6. No circular references
7. All node IDs are unique

## Phase 5: Rewrite Naming logic 
Target: rewrite it with Async post-order contrastive naming with one-time indexing

### 0) Initialize names
- Load the nested tree once (e.g., `tree = db.get_tree()`).
- Set initial in-memory names:
  - Category nodes (`node_type="category"`): `name = "node_id"`.
  - Paper nodes (`node_type="paper"`): treat as already named using `paper.summary` (or title+summary).

### 1) One-pass traversal to build lookups
Traverse the tree exactly once and build:
- `level_lookup[level] -> [node_id | paper_id]` (depth → items at that depth).
- `node_lookup[node_id] -> {"children": [node_id | paper_id], "parent": parent_node_id | None}`.
- Optional but recommended:
  - `node_ref[node_id] -> node_dict` (direct pointer to in-memory node for O(1) updates).

### 2) Async readiness via futures
Maintain naming state:
- `name_value[id] -> str | None` (paper: summary; category: LLM name).
- `name_future[id] -> asyncio.Future[str]`:
  - Paper futures are resolved immediately with the summary.
  - Category futures resolve when the node is named.

### 3) Post-order recursive naming (root launches, children first)
Implement `name_node_recursive(node_id)`:
1) Spawn/await `name_node_recursive(child_node_id)` for all category-children.
2) Await readiness of:
   - all children names (`await gather(name_future[child] ...)`)
   - all siblings’ children names (contrastive context).
3) Build the prompt via `_get_prompt("node_naming", ...)` and call the LLM (with retries).
4) On success:
   - Update in-memory node name (`node_ref[node_id]["name"] = new_name`),
   - Persist to DB (`db.update_tree_node_name(node_id, new_name)`),
   - Resolve `name_future[node_id]` so parents can proceed.

### 4) Concurrency control
- Wrap LLM calls with an `asyncio.Semaphore` to cap concurrent requests.
- Keep retry + fallback naming, but always resolve the node future on success/final fallback.


## Phase 6: Tree visualization
Target: to ensure the tree is properly displayed in frontend
- following the hierarchical top-down format
- no circular references, no shared parents of the same node, no mix of category/paper nodes under the same parent
- automatically spaced out the nodes in the same level to avoid overlapping
- automatically draw the edges in the tree in professional manner, if not already defaulted to more professional arrangment in the visualization package, my preference will be 
  - each parent will be horizontally centered among all its children nodes, but vertically one row above
  - all category nodes' edges will come out from the bottom center of the parent node, and landed in the top center of each child category node
- whenever a new paper (or a batch of new papers) are ingested, the clustering & naming will be redone, and tree visualization adjusted automatically

### Bug Fix: Paper nodes not displayed in tree diagram

**Root Cause:** In `page.tsx`, the tree layout logic used `buildCategoryTree()` to create a filtered tree containing only category nodes (for D3 hierarchical layout). However, when iterating over category nodes to attach paper children, the code accessed `catNode.data.children` - which came from the *filtered* category tree that had already excluded paper nodes. This meant `paperChildren` was always empty.

**Fix:** Added a lookup map `originalNodeById` that maps `node_id` to the original taxonomy node (which preserves all children including papers). When finding paper children for each category, the code now looks up the original node via `originalNodeById[catId]` instead of using the filtered `catNode.data`.

```typescript
// Build lookup map from node_id to original taxonomy node (with paper children)
const originalNodeById: Record<string, PaperNode> = {};
const buildLookup = (node: PaperNode) => {
  const nodeId = getNodeId(node);
  originalNodeById[nodeId] = node;
  for (const child of node.children || []) {
    buildLookup(child);
  }
};
buildLookup(taxonomy);

// Later, when finding paper children:
const originalNode = originalNodeById[catId];  // Use original, not filtered
const children = originalNode.children || [];
const paperChildren = children.filter((c) => isPaperNode(c));
```

## Phase 7: fixes

- there is now a button to "collapse pane", however, can we change it to some arrow button in the middle of the vertical separation line betwen the tree diagram on the left, and the pane on the right? I can toggle the on/off the right pane by clicking such arrow
- then tree diagram currently still spans across the entire window width, can we only display it on the left-hand-size of the window, and put the horizontal scroll bar into the tree-diagram, so that it will only scroll the tree diagram instead of the whole window
- please help me check the existing 'abbreviation' function, which was automatically triggered during paper ingestion. It was able to condense down paper names to the most prominent features of the paper, like a word or phrase. Currently, the rephrasing resulted in almost the entire paper title. Once we restore the original abbreviation capability, can we also persist it into the database, and use them as the display names for each paper node in the tree diagram?