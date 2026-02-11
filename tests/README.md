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
- `tests/storage/paperqa_index/` - Test paper indices
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

# Remaining issues
1. help me troubleshoot, do we still have timout issue during the test? if so, which tests? what's the root cause? are we using sync or async request to the vLLM endpoints (for both llm and embedding model)?

2.have we managed to test the ingestion for both local directory and slack channel?

3. if 2 is true, how did we manage to pass the local directory ingestion, when we only have 2 papers in tthe `test/storage/downloads`? Why you didn't put in more papers in the first place? why there are duplicated "attention is all you need" paper their with different titles