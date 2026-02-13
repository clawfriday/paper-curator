# Bug Log

## 1. NUL bytes in backend.log (file corruption)

**Symptom**: `logs/backend.log` contained 298,162 NUL (`\x00`) bytes on a single line, inflating the file and breaking log parsers.

**Root cause**: Truncating a log file (`> logs/backend.log`) while the backend Singularity container still has the file open. The OS fills the gap between the new file start and the old write offset with NUL bytes.

**Fix**: Cleaned with `tr -d '\0'`. No code change needed; avoid truncating logs while the container is running — restart the service instead.

---

## 2. Paper saved without embedding (invisible in UI)

**Symptom**: Paper `2510.18234` existed in the `papers` table but had `embedding = NULL` and no topic assignment, making it invisible in the tree and UI.

**Root cause**: The ingestion flow calls `/summarize` (which runs RAG) before `/papers/save`. Since the paper record doesn't exist yet, `rag_answer_async()` runs in ephemeral mode — chunks and embeddings are computed but discarded. `/papers/save` then creates the DB row but never computes an embedding.

**Fix**: Modified `/papers/save` in `src/backend/app.py` to call `rag.index_paper_async()` immediately after `db.create_paper()`, ensuring the embedding is always computed when a `pdf_path` is provided.

---

## 3. Frontend proxy timeout on arXiv endpoints (30s undici default)

**Symptom**: Ingesting paper `2601.20552` failed at "Resolving arXiv metadata" with HTTP 500 after exactly 30 seconds. Frontend log showed `socket hang up` / `ECONNRESET`.

**Root cause**: Next.js `rewrites()` proxy uses `fetch` (undici) with a default 30-second timeout. The arXiv API sometimes takes longer than 30s due to rate-limiting.

**Fix**: Created custom Next.js API routes (`src/frontend/src/app/api/arxiv/resolve/route.ts` and `download/route.ts`) with extended timeouts (120s and 300s respectively). Removed the `/api/arxiv/:path*` rewrite rule.

---

## 4. Re-categorize timeout (undici 300s headersTimeout + N+1 DB queries)

**Symptom**: "Re-categorize" failed with HTTP 500 after exactly 5 minutes. Frontend log showed `UND_ERR_HEADERS_TIMEOUT`.

**Root cause (layer 1 — proxy)**: `export const maxDuration` in Next.js API routes is Vercel-only and has no effect in standalone Node.js. The actual timeout comes from undici's hardcoded 300-second `headersTimeout`. Classify with 2000+ papers takes 8+ minutes.

**Root cause (layer 2 — N+1 queries)**: `naming.py`'s `TreeIndex.__init__()` called `db.get_paper_by_id()` individually for every paper node (2000+ times), each creating a new PostgreSQL connection and fetching `SELECT *` (including the 4096-float embedding vector). This alone took 12+ minutes and blocked the event loop.

**Fix (proxy)**: Created `src/frontend/src/lib/backend-proxy.ts` using `node:http` instead of `fetch` — no undici headersTimeout limit. Updated classify, rebalance, and batch-ingest routes to use it.

**Fix (N+1)**: Added `db.get_papers_lightweight()` for single-query bulk loading without embeddings. Updated `naming.py` to accept a pre-loaded `paper_cache` dict. Added `db.get_papers_missing_embeddings()` to avoid loading all embeddings just to check which are NULL.

---

## 5. Topic query numpy serialization error

**Symptom**: `GET /topic/check?topic_query=learning%20dynamics` returned HTTP 500 with `PydanticSerializationError: Unable to serialize unknown type: <class 'numpy.ndarray'>`.

**Root cause**: `db.get_topics_by_query()` used `SELECT t.*` which includes the `embedding` column. pgvector registers a custom type adapter that returns embeddings as `numpy.ndarray`. Pydantic/FastAPI cannot serialize numpy arrays to JSON. Same issue in `db.get_topic_by_id()` which explicitly selected `t.embedding`.

**Fix**: Changed `get_topics_by_query()`, `get_topic_by_id()`, and `get_topic_by_name()` in `src/backend/db.py` to use explicit column lists excluding `embedding`. The embedding vector is not needed for any topic metadata response.
