# Paper Curator

An AI-powered research paper management system that automatically organizes, summarizes, and enables semantic search across academic papers.

## How It Works

### Core Features

1. **Paper Ingestion**
   - Import papers via arXiv ID, PDF upload, or Slack channel
   - Automatic metadata extraction (title, authors, abstract)
   - PDF indexing for semantic search using PaperQA2

2. **Hierarchical Categorization**
   - Embedding-based clustering using paper abstracts
   - Automatic tree structure generation (k-means with silhouette scoring)
   - LLM-generated category names based on cluster content

3. **Paper Summarization**
   - Structured analysis: method, benefits, rationale, results
   - RAG-based Q&A on paper content
   - Cached indexes for fast repeated queries

4. **External Data Integration**
   - GitHub repository lookup (Papers With Code + GitHub Search)
   - Reference extraction via Semantic Scholar
   - Similar paper recommendations

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │────▶│   Backend   │────▶│  PostgreSQL │
│  (Next.js)  │     │  (FastAPI)  │     │  (pgvector) │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌─────────┐   ┌──────────┐
              │   LLM   │   │ External │
              │ Endpoint│   │   APIs   │
              └─────────┘   └──────────┘
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL with pgvector extension
- OpenAI-compatible LLM and embedding endpoints

### Configuration

Create `config/config.yaml`:

```yaml
server:
  frontend_port: 3000
  backend_port: 3100

endpoints:
  llm_base_url: "http://localhost:8001/v1"      # LLM endpoint
  embedding_base_url: "http://localhost:8002/v1" # Embedding endpoint
  api_key: "your-api-key"

ingestion:
  skip_existing: false  # Set true to skip already-ingested papers

classification:
  branching_factor: 5   # Max children per tree node
  rebuild_on_ingest: true  # Auto-rebuild tree after ingestion
```

---

### Local Setup (Docker)

```bash
# Clone repository
git clone <repo-url>
cd paper-curator

# Start all services
docker compose -f src/compose.yml up --build

# Access UI at http://localhost:3000
```

---

### HPC Setup (Singularity)

For HPC environments without Docker:

```bash
# 1. Build containers (requires fakeroot)
make singularity-build

# 2. Start services
make singularity-run

# 3. Check status
./scripts/hpc-services.sh status

# 4. Stop services
make singularity-stop
```

**Required directory structure:**
```
paper-curator/
├── containers/      # Built .sif files
├── config/          # config.yaml
├── storage/         # PDFs and indexes (auto-created)
├── pgdata/          # PostgreSQL data (auto-created)
├── logs/            # Service logs (auto-created)
└── run/             # PID files (auto-created)
```

**Slack token setup** (for Slack ingestion):
```bash
# Store your Slack User OAuth Token
echo "xoxp-your-token" > ~/.ssh/.slack
chmod 600 ~/.ssh/.slack
```

---

## Usage

### Web UI

Access at `http://localhost:3000`

- **Ingest**: Add papers via arXiv URL, PDF, or Slack channel
- **Tree View**: Browse hierarchical categorization
- **Paper Details**: View summary, Q&A, references, similar papers
- **Re-classify**: Rebuild tree structure from current papers

### Command Line

```bash
# Ingest from default Slack channel
make pull-slack

# Ingest from custom channel
make pull-slack SLACK_CHANNEL=https://app.slack.com/client/WORKSPACE/CHANNEL

# Check service health
curl http://localhost:3100/health

# Trigger re-classification
curl -X POST http://localhost:3100/papers/classify

# Query paper count
singularity exec instance://paper-curator-db \
  psql -h localhost -p 5432 -U curator -d paper_curator \
  -c "SELECT COUNT(*) FROM papers;"
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /papers/batch-ingest` | Bulk import from directory or Slack |
| `POST /papers/classify` | Rebuild tree with clustering |
| `POST /summarize` | Generate paper summary |
| `POST /qa` | Ask question about a paper |
| `GET /tree` | Get hierarchical tree structure |
| `POST /papers/similar` | Find similar papers |

See `src/backend/README.md` for full API documentation.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `not_in_channel` error | Invite Slack bot to channel or use User Token |
| `Read-only file system` | Ensure `storage/` directory is writable |
| Papers not in tree | Click "Re-classify" to rebuild tree |
| Database connection failed | Check PostgreSQL is running on port 5432 |

### Logs

```bash
# Backend logs
tail -f logs/backend.log

# Frontend logs
tail -f logs/frontend.log

# Database status
./scripts/hpc-services.sh status
```

---

## License

MIT
