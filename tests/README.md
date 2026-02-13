# Paper Curator Tests

Tests are organized into three categories based on their requirements and scope.

## Test Structure

```
tests/
├── README.md                 # This file
├── conftest.py              # Shared pytest fixtures
├── functional/              # Single-capability tests (require services)
│   └── README.md
├── validation/              # Connectivity + input validation tests
│   └── README.md
├── integration/             # End-to-end workflow tests (test database)
│   └── README.md
└── storage/                 # Test data and isolated database
    ├── downloads/           # Git-tracked sample PDFs (10 papers)
    └── pgdata/              # Test database data directory (gitignored)
```

## Test Categories

| Category | Description | Backend Required | LLM Required | Database |
|----------|-------------|------------------|--------------|----------|
| **validation** | Connectivity checks + input validation (422 tests) | For connectivity | No | Production |
| **functional** | Single capability tests (download, extract, embed, etc.) | Yes | Yes | Production |
| **integration** | End-to-end workflows (ingest, cluster, query) | Yes | Yes | **Test DB (port 5433)** |

## Quick Start

```bash
# 1. Run validation tests first (fast, catches config issues)
pytest tests/validation/ -v

# 2. Run functional tests (requires backend + LLM)
BACKEND_URL=http://localhost:3100 pytest tests/functional/ -v -s

# 3. Run integration tests (requires test database)
# First, start test database on port 5433 (see integration/README.md)
PGPORT=5433 pytest tests/integration/ -v -s --order-dependencies
```

## Test Database Setup

Integration tests use an isolated PostgreSQL instance to avoid affecting production data.

### Using Singularity (HPC)

```bash
# Start test database container
singularity instance start \
  --bind $(pwd)/tests/storage/pgdata:/var/lib/postgresql/data \
  --env POSTGRES_USER=curator \
  --env POSTGRES_PASSWORD=curator123 \
  --env POSTGRES_DB=paper_curator_test \
  --network-args "portmap=5433:5432/tcp" \
  containers/pgvector.sif pgtest

# Initialize schema
PGPORT=5433 PGDATABASE=paper_curator_test python scripts/init_db.py
```

### Using Docker

```bash
docker run -d \
  --name paper-curator-test-db \
  -p 5433:5432 \
  -v $(pwd)/tests/storage/pgdata:/var/lib/postgresql/data \
  -e POSTGRES_USER=curator \
  -e POSTGRES_PASSWORD=curator123 \
  -e POSTGRES_DB=paper_curator_test \
  pgvector/pgvector:pg16
```

## Environment Variables

```bash
# Backend URL (for functional tests)
BACKEND_URL=http://localhost:3100

# Test database (for integration tests)
PGHOST=localhost
PGPORT=5433
PGDATABASE=paper_curator_test
PGUSER=curator
PGPASSWORD=curator123

# Fail instead of skip when services unavailable
REQUIRE_EXTERNAL_ENDPOINTS=1
```

## Test Data

### Sample PDFs (git-tracked)

The following 10 PDFs in `tests/storage/downloads/` are committed to the repository:

| File | Paper | Domain |
|------|-------|--------|
| 1706.03762.pdf | Attention Is All You Need | NLP |
| 1810.04805.pdf | BERT | NLP |
| 2005.14165.pdf | GPT-3 | NLP |
| 1512.03385.pdf | ResNet | Vision |
| 1406.2661.pdf | GAN | Vision |
| 1706.03741.pdf | PPO | RL |
| 2010.11929.pdf | Vision Transformer | Vision |
| 2103.00020.pdf | CLIP | Multimodal |
| 2302.13971.pdf | LLaMA | NLP |
| 2303.08774.pdf | GPT-4 | NLP |

### Dynamic Test Data (gitignored)

- `tests/storage/pgdata/` - Test database files
- Any downloaded PDFs beyond the 10 tracked ones

## Writing New Tests

See individual README files in each subfolder for:
- Test case templates
- Success criteria definitions
- Mock data examples
- Fixture usage

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
test:
  steps:
    - name: Validation tests
      run: pytest tests/validation/ -v
      
    - name: Functional tests
      run: pytest tests/functional/ -v -s
      env:
        BACKEND_URL: http://localhost:3100
        
    - name: Integration tests
      run: |
        # Start test database
        docker-compose -f tests/docker-compose.test.yml up -d
        sleep 10
        PGPORT=5433 pytest tests/integration/ -v -s
```

## Recommended Additional Test Scope

Test gaps identified from production debugging. Each item describes a failure mode not caught by the current suite.

| # | Area | What to test | Why |
|---|------|-------------|-----|
| 1 | **Frontend proxy timeout** | Send requests to long-running endpoints through the frontend proxy (port 3000) and verify they succeed, not 500/504. | All existing tests hit the backend directly (port 3100). Undici's hardcoded 300s headersTimeout killed classify requests before the backend finished. |
| 2 | **Save-then-retrieve embedding** | After calling `/papers/save` with a valid pdf_path, query the paper's cached-data and assert `embedding` is non-null and response includes `indexed: true`. | A race condition caused papers to be saved without embeddings, making them invisible in clustering and the UI. |
| 3 | **NUL byte handling** | Extract text from every sample PDF and assert no `\x00` characters remain. | Some PDFs contain NUL bytes that PostgreSQL TEXT columns reject, silently failing the indexing step. |
| 4 | **Classify at scale** | Run `/papers/classify` after ingesting 10+ papers and verify it completes with `papers_classified > 0` and `nodes_named > 0`. | Re-categorize took 8+ minutes; prior tests only tested with a handful of papers and did not catch timeout or N+1 query issues. |
| 5 | **Config effect: rebuild_on_ingest** | Save a paper and assert `rebuild_triggered` matches the `rebuild_on_ingest` config value. | Setting `rebuild_on_ingest: true` caused full tree rebuilds on every save, saturating the CPU and hanging the server. |
| 6 | **Backend responsiveness** | Start a long-running operation (classify) in a background thread, then verify `/health` still responds within 10 seconds. | Synchronous DB calls in the naming module blocked the event loop, making the entire server unresponsive. |
| 7 | **Topic endpoint serialization** | Call `GET /topic/check`, `GET /topic/list`, and `GET /topic/{id}` and verify each returns valid JSON with HTTP 200 (or 404 for missing topics). | pgvector returns embedding columns as numpy.ndarray, which Pydantic cannot serialize. DB queries using `SELECT *` or explicitly including `embedding` cause HTTP 500 PydanticSerializationError. |

### Priority

| Priority | Items | Impact |
|----------|-------|--------|
| **P0** | 2, 7 | Silent data loss / user-facing 500 |
| **P1** | 1, 3, 6 | Timeout errors / silent indexing failure / server hang |
| **P2** | 4, 5 | Performance regressions / misconfiguration |
