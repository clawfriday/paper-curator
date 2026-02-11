# Integration Tests

Integration tests verify complete end-to-end workflows using an isolated test database. These tests simulate real user workflows: ingesting papers, querying, managing the tree, etc.

## Test Database Architecture

Tests use a separate PostgreSQL instance with its own data directory:

```
Production:                      Test:
┌─────────────────┐             ┌─────────────────┐
│ Port: 5432      │             │ Port: 5433      │
│ Data:           │             │ Data:           │
│ storage/pgdata  │             │ tests/storage/  │
│                 │             │   pgdata        │
└─────────────────┘             └─────────────────┘
```

Same container image, different port and data directory. No rebuild needed.

## Prerequisites

- Test database running on port 5433 (see setup below)
- Backend configured to use test database (PGPORT=5433)
- LLM and embedding endpoints accessible
- Sample PDFs in tests/storage/downloads/

## Test Database Setup

Start test database container:

```bash
# Using Singularity (HPC)
singularity instance start \
  --bind tests/storage/pgdata:/var/lib/postgresql/data \
  --env POSTGRES_USER=curator \
  --env POSTGRES_PASSWORD=curator123 \
  --env POSTGRES_DB=paper_curator_test \
  containers/pgvector.sif pgtest

# Wait for startup, then initialize
PGPORT=5433 python scripts/init_db.py

# Or using Docker
docker run -d \
  --name paper-curator-test-db \
  -p 5433:5432 \
  -v $(pwd)/tests/storage/pgdata:/var/lib/postgresql/data \
  -e POSTGRES_USER=curator \
  -e POSTGRES_PASSWORD=curator123 \
  -e POSTGRES_DB=paper_curator_test \
  pgvector/pgvector:pg16
```

## Running Tests

```bash
cd tests/integration

# Set environment for test database
export PGPORT=5433
export PGDATABASE=paper_curator_test

# Run all integration tests (in order)
pytest -v -s --order-dependencies

# Run specific test
pytest -v -s test_integration.py::test_ingest_single_paper
```

## Test Execution Order

Tests must run in order due to dependencies:

1. test_ingest_single_paper (creates base paper)
2. test_paper_remove (removes it)
3. test_ingest_local_folder (ingests multiple papers)
4. test_clustering (builds tree)
5. test_rename_all (renames categories)
6. test_category_rename (renames specific category)
7. test_ingest_channel (tests Slack ingestion)
8. test_topic_query_integration (tests topic query workflow)

---

## Test Cases

### test_ingest_single_paper

Ingest "Attention Is All You Need" (arxiv:1706.03762) and verify complete processing.

Inputs:
- arxiv_id: "1706.03762"
- Use /papers/save endpoint after resolving metadata

Verification steps:

1. Paper PDF extraction:
   - PDF downloaded to storage/downloads/
   - Text extracted successfully (length > 5000 chars)
   - No extraction errors

2. Metadata extraction:
   - ArXiv metadata present: title, authors, abstract, published_at
   - Semantic Scholar metadata present (if available): citation_count, influential_citation_count
   - All required fields non-empty

3. Embeddings:
   - Chunk-wise embeddings generated (check index file exists)
   - Paper-level embedding stored in database (check papers.embedding column)
   - Embedding dimension matches expected (768 or 1536)

4. Summary:
   - Summary generated and stored in papers.summary
   - Summary length > 200 characters
   - Summary mentions key concepts from paper

5. Tree node:
   - Paper node created in tree with correct paper_id
   - Node has valid parent (category or root)
   - Node name matches paper abbreviation

6. Database records:
   - Paper record exists in papers table
   - All foreign key relationships valid
   - Created_at timestamp set

---

### test_paper_remove

Remove the "Attention Is All You Need" paper and verify cleanup.

Inputs:
- arxiv_id: "1706.03762"
- DELETE /papers/{arxiv_id}

Verification steps:

1. API response:
   - Status 200
   - Response indicates successful deletion

2. Database cleanup:
   - Paper record removed from papers table
   - Related records removed (cached_repos, references, etc.)

3. Tree cleanup:
   - Paper node removed from tree
   - Parent category still exists (not orphaned)

4. File cleanup:
   - PaperQA index file removed (storage/paperqa_index/{arxiv_id}.pkl)
   - PDF file optionally removed or retained based on config

---

### test_category_rename

Create a test category, rename it, and verify the update.

Inputs:
- Create category with initial name "Test Category"
- Add 2-3 papers to category
- POST /categories/rename with LLM-generated name

Verification steps:

1. Category creation:
   - Category node added to tree
   - Papers assigned as children

2. Rename operation:
   - Status 200
   - New name returned in response
   - New name reflects papers' topics (LLM-generated)

3. Tree update:
   - Category node has updated name
   - Children (papers) still attached
   - No orphaned nodes

---

### test_ingest_local_folder

Ingest all PDFs from tests/storage/downloads/ folder.

Inputs:
- directory: "tests/storage/downloads"
- POST /papers/batch-ingest

Verification steps:

1. Ingestion count:
   - Number of successfully ingested papers matches PDF count in folder
   - Progress log shows each paper processed

2. Database consistency:
   - Paper count in papers table matches ingested count
   - Each paper has: arxiv_id, title, authors, pdf_path

3. Tree consistency:
   - Number of paper nodes in tree matches paper count
   - All papers have corresponding tree nodes

4. Specific paper check ("Attention Is All You Need"):
   - Use same verification as test_ingest_single_paper
   - All fields correctly populated

---

### test_ingest_channel

Ingest papers from a Slack channel (limited to 10).

Inputs:
- slack_channel: configured test channel URL
- slack_token: from ~/.ssh/.slack
- limit: 10 (if supported, otherwise process all and verify first 10)
- POST /papers/batch-ingest

Verification steps:

1. Ingestion count:
   - At least 1 paper ingested (channel may have varying content)
   - No more than expected limit

2. Database consistency:
   - Paper count in papers table matches ingested count
   - Each paper has valid arxiv_id

3. Tree consistency:
   - Number of paper nodes matches paper count
   - No duplicate paper nodes

Note: This test may be skipped if Slack token unavailable or channel empty.

---

### test_clustering

Verify tree is successfully built after ingestion.

Inputs:
- POST /papers/classify (triggers full tree rebuild)

Verification steps:

1. API response:
   - Status 200
   - Processing completed without timeout

2. Tree structure:
   - Root node exists with node_id "root"
   - At least one category created
   - All papers assigned to categories (no orphans at root level unless intended)

3. Category quality:
   - Categories have meaningful names (not just "Category 1")
   - Papers in same category are thematically related (spot check)

4. Tree integrity:
   - No duplicate node_ids
   - No cycles
   - All paper nodes have valid paper_id references

---

### test_rename_all

Verify all category nodes are correctly renamed.

Inputs:
- POST /papers/reabbreviate-all or equivalent batch rename

Verification steps:

1. API response:
   - Status 200
   - Count of renamed categories returned

2. Category names:
   - All category nodes have non-empty names
   - Names are descriptive (not generic like "Untitled")
   - Names reflect contained papers' topics

3. No side effects:
   - Paper nodes unchanged
   - Tree structure unchanged
   - Paper abbreviations unchanged (unless explicitly updated)

---

### test_topic_query_integration

Complete topic query workflow test.

Inputs:
- topic: "transformer attention mechanisms"
- question: "What are the key innovations in attention mechanisms across these papers?"

Workflow steps:

1. Topic search:
   - POST /topic/search with topic
   - Verify papers returned with similarity scores

2. Topic creation:
   - POST /topic/create with name and topic_query
   - Verify topic_id returned

3. Add papers:
   - POST /topic/{id}/papers with paper_ids from search
   - Verify papers added successfully

4. Query topic:
   - POST /topic/{id}/query with question
   - Verify answer synthesized from multiple papers

5. Cleanup:
   - DELETE /topic/{id}
   - Verify topic removed

Verification for query response:
- answer field present and length > 100 chars
- papers_queried count matches added papers
- successful_queries > 0
- Answer references multiple papers (cross-paper synthesis)

---

## Fixtures

### conftest.py for integration tests

```python
import pytest
import os

@pytest.fixture(scope="module")
def test_db_connection():
    """Ensure test database is accessible."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5433")),
        database=os.environ.get("PGDATABASE", "paper_curator_test"),
        user=os.environ.get("PGUSER", "curator"),
        password=os.environ.get("PGPASSWORD", "curator123"),
    )
    yield conn
    conn.close()

@pytest.fixture(scope="module")
def clean_test_db(test_db_connection):
    """Clean test database before test module."""
    with test_db_connection.cursor() as cur:
        cur.execute("TRUNCATE papers, tree_nodes, topics, topic_papers CASCADE")
    test_db_connection.commit()
    yield
    # Optionally clean after tests too

@pytest.fixture
def sample_pdfs_dir():
    """Path to test PDFs directory."""
    return "tests/storage/downloads"
```

---

## Test Data

### Git-tracked PDFs in tests/storage/downloads/

The following 10 PDFs should be committed to the repository for reproducible testing:

1. 1706.03762.pdf - Attention Is All You Need (Transformer)
2. 1810.04805.pdf - BERT
3. 2005.14165.pdf - GPT-3
4. 1512.03385.pdf - ResNet
5. 1406.2661.pdf - GAN
6. 1706.03741.pdf - Proximal Policy Optimization (PPO)
7. 2010.11929.pdf - Vision Transformer (ViT)
8. 2103.00020.pdf - CLIP
9. 2302.13971.pdf - LLaMA
10. 2303.08774.pdf - GPT-4 Technical Report

These cover diverse ML domains: NLP, Vision, RL, Multimodal - enabling meaningful clustering tests.
