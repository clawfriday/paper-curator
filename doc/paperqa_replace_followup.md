# PaperQA Replacement Follow-up (based on current `hawkoli/main`)

## What changed (verified)

- PaperQA dependency is removed from backend runtime path.
- New custom RAG pipeline is in `src/backend/rag.py`:
  - PDF extract via `pymupdf`
  - chunking + overlap
  - embedding via OpenAI-compatible endpoint
  - chunk persistence in `paper_chunks` (pgvector)
  - retrieval via DB vector search or in-memory fallback
- DB schema includes `paper_chunks` and uses `vector(4096)`.

## Main risk areas

1. **Vector dimension lock-in (4096)**
   - If new embedding endpoint returns non-4096 vectors, writes/search will fail.
2. **Chunking defaults may hurt recall/latency tradeoff**
   - `chunk_chars=5000` can be too coarse for some QA prompts.
3. **Cold-start cost**
   - First query on non-indexed paper does extraction+embedding; can feel slow.
4. **Ephemeral fallback consistency**
   - Behavior differs between DB-backed retrieval and in-memory fallback.
5. **Topic query fan-out cost**
   - Multi-paper query can produce high LLM token/latency usage.
6. **Frontend container build fragility**
   - Missing `public/` in repo can break image build (patched in Dockerfile).

## Simple verification plan

1. **Health + startup**
   - `GET /health` = 200
2. **Single paper ingest path**
   - Ingest one PDF (manual Slack pull by user)
   - confirm `papers.embedding` is set and `paper_chunks` has rows
3. **QA correctness smoke**
   - ask 2 factual questions from that paper
   - ensure answers reference paper-specific details
4. **Structured QA + topic query smoke**
   - run one structured QA and one topic query with 3+ papers
   - ensure no 5xx and reasonable latency
5. **Restart persistence**
   - restart backend/frontend containers
   - verify previously indexed papers still answer without re-embed

## Mitigation steps

1. **Dimension guardrail**
   - add startup check: embedding dimension must match DB (`4096`) or fail fast.
2. **RAG tunables profile**
   - keep a "fast" and "quality" preset for chunk/evidence knobs.
3. **Timeout/retry wrappers**
   - bounded retry for embedding/LLM calls; return actionable 503 errors.
4. **Query budget control (topic query)**
   - cap max papers and per-paper retrieved chunks in config.
5. **Operational runbook**
   - document re-index flow when embedding endpoint/model changes.
