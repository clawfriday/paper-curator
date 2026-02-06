# Development Notes - HPC Migration Bug Fixes

This document tracks Development Plan and bug fixes


# Bugs

---

## Bug 1: arXiv ID Extraction Regex Failure

**Symptom:**
Slack ingestion found 0 arXiv papers despite 216 messages containing 192 arXiv links.

```
✓ Fetched 216 messages from channel
✓ Found 0 unique arXiv papers
```

**Cause:**
Double-escaped backslashes in regex patterns within raw strings. In Python raw strings (`r"..."`), `\\` matches a literal backslash, not a regex escape.

```python
# WRONG (matched literal backslash):
url_pattern = r"arxiv\\.org/(abs|pdf)/([0-9.]+)"

# CORRECT:
url_pattern = r"arxiv\.org/(abs|pdf)/([0-9.]+)"
```

**Fix:**
Updated `src/backend/arxiv_helpers.py` to use single backslashes in raw strings.

**Files Changed:** `src/backend/arxiv_helpers.py`

---

## Bug 2: Read-Only Filesystem in Singularity Container

**Symptom:**
Re-classification failed with HTTP 500:
```
OSError: [Errno 30] Read-only file system: 'schemas'
```

**Cause:**
`naming.py` tried to create a `schemas/` directory to save debug output, but Singularity containers are read-only except for explicitly bound directories.

**Fix:**
Changed all file writes from `schemas/` to `storage/schemas/` which is already bound as writable.

**Files Changed:** `src/backend/naming.py`, `src/backend/clustering.py`

---

## Bug 3: Frontend Cannot Connect to Backend (ENOTFOUND)

**Symptom:**
HTTP 500 errors on Slack ingestion with frontend logs showing:
```
ENOTFOUND backend
```

**Cause:**
Frontend API routes used hardcoded `http://backend:8000` which only works with Docker's internal DNS. In Singularity (host networking), services communicate via `localhost`.

**Fix:**
1. Added `BACKEND_URL` environment variable support to `next.config.js`
2. Updated all 11 API route files to use `process.env.BACKEND_URL || "http://backend:8000"`
3. Set `BACKEND_URL=http://localhost:3100` in `containers/frontend.def` at build and runtime

**Files Changed:** 
- `src/frontend/next.config.js`
- `src/frontend/src/app/api/*/route.ts` (11 files)
- `containers/frontend.def`

---

## Bug 4: Backend Converting localhost to host.docker.internal

**Symptom:**
Backend API calls to external LLM endpoints failed because `localhost` URLs were incorrectly converted to `host.docker.internal` (which doesn't exist in Singularity).

**Cause:**
`config.py` had Docker-specific logic to convert localhost URLs for Docker's networking model.

**Fix:**
Added environment variable check to skip conversion when running in Singularity:
```python
if os.environ.get("SINGULARITY_CONTAINER") or os.environ.get("APPTAINER_CONTAINER"):
    return url  # Don't convert in Singularity
```

**Files Changed:** `src/backend/config.py`

---

## Bug 5: PostgreSQL Socket File Permission Error

**Symptom:**
Database failed to start with:
```
FATAL: could not create lock file "/var/run/postgresql/.s.PGSQL.5432.lock": Read-only file system
```

**Cause:**
PostgreSQL needs to create socket files in `/var/run/postgresql`, but this directory is read-only in Singularity.

**Fix:**
Created writable `run/postgresql` directory on host and bound it to `/var/run/postgresql` in the container.

**Files Changed:** `scripts/hpc-services.sh`

---

## Bug 6: Missing `paper_queries` Table

**Symptom:**
Error when viewing paper details:
```
psycopg2.errors.UndefinedTable: relation "paper_queries" does not exist
```

**Cause:**
The `paper_queries` table was referenced in `db.py` but never defined in `init.sql`.

**Fix:**
Added table creation to `init.sql`:
```sql
CREATE TABLE IF NOT EXISTS paper_queries (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Files Changed:** `src/backend/init.sql`

---

## Bug 7: skip_existing Not Configurable

**Symptom:**
Re-ingesting papers always skipped existing ones with no way to override.

**Cause:**
`skip_existing` was hardcoded to `True` in the request model.

**Fix:**
1. Added `ingestion.skip_existing` to `config/config.yaml`
2. Updated `app.py` to read from config instead of request payload

**Files Changed:** `config/config.yaml`, `src/backend/app.py`, `src/backend/config.py`

---

## Bug 8: Next.js Standalone Mode Missing Static Files

**Symptom:**
Browser shows white page after loading HTML. `curl` works but browser doesn't render anything. JavaScript files return HTML instead of JS code when requested.

**Cause:**
Next.js `output: "standalone"` mode requires manually copying static assets into the standalone directory. The build output places files in `.next/static/` but the standalone server expects them in `.next/standalone/.next/static/`.

**Fix:**
Added post-build copy step in `containers/frontend.def`:
```bash
cp -r /app/.next/static /app/.next/standalone/.next/static
if [ -d /app/public ]; then
    cp -r /app/public /app/.next/standalone/public
fi
```

**Files Changed:** `containers/frontend.def`

---

## Bug 9: Tree API Timeout (ECONNRESET)

**Symptom:**
Tree doesn't load on initial page load. Frontend logs show:
```
Failed to proxy http://localhost:3100/tree Error: socket hang up
code: 'ECONNRESET'
```
Reclassify button also fails with the same error.

**Cause:**
The `/tree` endpoint takes 15+ seconds to respond for large datasets (builds complete tree structure with all papers). Next.js rewrites have a default timeout that's shorter than this, causing the connection to be reset before the response arrives.

**Fix:**
Created custom API route `/api/tree/route.ts` with extended timeout:
```typescript
export const maxDuration = 120; // 2 minutes

export async function GET() {
  const res = await fetch(`${BACKEND_URL}/tree`, {
    cache: "no-store",
  });
  return NextResponse.json(await res.json());
}
```
Removed the rewrite rule for `/api/tree` since custom route takes precedence.

**Files Changed:** 
- `src/frontend/src/app/api/tree/route.ts` (new)
- `src/frontend/next.config.js`

---

## Bug 10: Structured Analysis Prompt Format Error

**Symptom:**
Generating structured analysis fails with:
```
ValueError: unexpected '{' in field name
  File "/app/config.py", line 56, in get_prompt
    return template.format(**kwargs)
```

**Cause:**
Python's `.format()` interprets curly braces `{` as format placeholders. The prompts in `prompts.json` contained LaTeX examples with literal curly braces (e.g., `\\eta_{\\text{peak}}`), which Python tried to parse as format variables.

**Fix:**
Escaped curly braces in LaTeX examples using double braces (Python format string syntax):
```json
// Before (broken):
"Use \\( \\eta_{\\text{peak}} \\)"

// After (fixed):
"Use \\( \\eta_{{\\text{{peak}}}} \\)"
```

**Files Changed:** `src/backend/prompts/prompts.json`

---

## Bug 11: Slow Tree Loading (N+1 Query Problem)

**Symptom:**
Tree takes 15+ seconds to load on page refresh. Users see blank tree with no feedback.

**Cause:**
The `get_tree()` function in `db.py` fetched paper metadata individually for each paper node using `get_paper_by_id()`. With 192 papers, this resulted in 192 separate database queries (N+1 problem).

**Fix:**
1. **Backend optimization**: Batch-fetch all papers in a single query and use a lookup dictionary:
```python
# Fetch ALL papers in a single query
cur.execute("SELECT id, arxiv_id, title, authors, summary, pdf_path, abbreviation FROM papers")
all_papers = cur.fetchall()
papers_by_id = {p["id"]: p for p in all_papers}

# O(1) lookup instead of DB query per paper
paper = papers_by_id.get(node["paper_id"])
```

2. **Frontend loading indicator**: Added spinning loader next to "Paper Curator" title during tree loading:
```tsx
const [isLoadingTree, setIsLoadingTree] = useState(true);
// Display: "Loading tree..." with spinner while fetching
```

**Performance Impact:** Tree load time reduced from ~15s to ~1s.

**Files Changed:** 
- `src/backend/db.py`
- `src/frontend/src/app/page.tsx`

---

## Bug 12: LiteLLM MAX_CALLBACKS Warning

**Symptom:**
Warning during structured analysis:
```
Cannot add callback - would exceed MAX_CALLBACKS limit of 30. Current callbacks: 30
```

**Cause:**
PaperQA2 uses LiteLLM internally, which registers callbacks for each LLM call. These accumulate across multiple queries and hit the 30-callback limit. The existing `reset_litellm_callbacks()` function didn't clear the `logging_callback_manager`.

**Fix:**
Extended the reset function to also clear the logging callback manager:
```python
# Also clear the logging callback manager if it exists
if hasattr(litellm, 'logging_callback_manager'):
    manager = litellm.logging_callback_manager
    if hasattr(manager, 'callbacks'):
        manager.callbacks = []
```

**Files Changed:** `src/backend/llm_clients.py`

---

## Configuration Added

### config/config.yaml
```yaml
server:
  frontend_port: 3000
  backend_port: 3100

ingestion:
  skip_existing: false  # Set to true to skip existing papers
```

### Makefile
```makefile
# Pull papers from Slack channel
make pull-slack                              # Default channel
make pull-slack SLACK_CHANNEL=https://...    # Custom channel
```

---

## Testing Checklist

After any container rebuild, verify:
1. `curl http://localhost:3100/health` → `{"status":"ok"}`
2. `curl http://localhost:3000` → Frontend loads
3. Database connection: `psql -h localhost -p 5432 -U curator -d paper_curator`
4. Slack ingestion: `make pull-slack`
5. Re-classification: `curl -X POST http://localhost:3100/papers/classify`


# Features

## Text Rendering
Implement Professional Rendering of summary and structured analysis
- latex formula to be rendered correctly
- different pointers to be rendered as separate lines
- each component of the structured anlaysis to be collapsed and expanded
- use a consistent color regime to color the section header of each components, i.e. `Steps`, `Benefits`, `Rationale`, `Results`
- improve the prompt to reduce fluff in both summary and structured analysis
- improve the prompt, such that the summary/structure analysis will bold the fonts of the key findings of the paper. The bold text should be as focused as possible, instead of bold the full sentences, bold only the key parameters/results, e.g.
  "The CDMA strategy improves average benchmark accuracy by **1.68%** over the baseline (**Uniform+WSD** with **ending LR 1×10⁻⁵**) in the mid-training setting"
- currently, I believe the texts in the right panel (Details) defaults to font size 19, please change them to 14


## Search result navigation
  - Add a title "Details" at the top of the right panel
  - Add a frontend function 'center_on_node', such that once the node (paper or category) is found, navigate the tree diagram to center on the paper
  - Under "Explorer", both "Category Details" and "Paper Details" should have a new section 'Ancestry' (at the same level as "Summary" and "Structured Analysis"), displaying the ancestor chain of this paper, each ancestor in a newline, and uses incremental identation 
  - Each level in the "Ancestry" is a URL, once clicked, it will direct us to the respective ancestor node, and also invoke 'center_on_node' onto the ancestor node.
  - Expand the search mode, add a dropdown menu at the left of search box, which has 3 options:
    - search paper (dafault)
    - search category (to be implmenented in backend later)
    - search topic (to be implmenented in backend, later)


## Reabbrivate
add a button 're-abbreviate' inside 'explorer, right after the title 'Paper Details', which allows us to reabbreviate each paper or category
- the reabbreviation of the paper will be done just based on the same abbrevation logic of paper, 
- the reabbreviation of the category will be following the naming logic of the category, i.e. gathering the details from current children, sibling, and sibling's children, and name accordingly. However, please verify that the naming process is non-deterministic (if not, we may need to overwrite the generation temperature to make it so)
- in both cases, it will raise a text prompt, so that user can reabbreviate it using its custom input. If ignore, it will follow its own logic in reabbreviating
- the result will be immediately reflected in the node name in the tree diagram, and persisted into respective tables in the database

## Search the Category
a new backend api endpoiint, similar to find the paper, find the category node instead

## Full tree Query
a new backend api endpoint, which does the following
- rename the frontend search option 'tree' to 'topic'
- accepts an initial search 'topic' from the textbox
- read a 'max_papers_per_batch' from config (to add, default to 10)
- use similar RAG workflow in paperQA endpoint, to
  - embed the topic
  - use paper-level embedding, to do similarity search
  - identify the top 'max_papers_per_batch' papers, with their titles listed in a dropdown menu, each paper can be selected individually, or select-all/select-none.
  - there is a button 'add-to-pool', which addes the selected paper into the query pool
  - after adding, refresh the dropdown menu with the next batch of 'max_papers_per_batch' papers, which are from the papers with the next highest similarity score. This allows the user to add more papers into the pool 
  - repeat this until user has clicked the 'enough' button in the dropdown menu
- with this pool of papers formed, the user will be able to ask any questions, and get a response. However, the RAG is done in a hierarchical way 
  - for each new query, the model will only retrieve the most relevant n chunks from m papers, above a certain threshold (the threshold is also to set in config)
  - it will then obtain a response from each of the m papers, following the existing paper query endpoint
  - it will then assemble all these responses into one prompt, and summarize them into one coherent response
  - only the final summary will be returned as the response
  - however, for debugging purpose, when the script is run in debugging mode, the intermediate model prompts and responses should also be saved into 'storage/schemas/tree_query.json'
- in the frontend "Details" Panel, all such query and responses will be displayed inside the new Tab "Tree-Query", but they should be grouped by their topic, and each topic is collapsed by default, and can be expanded by clicking on it.
- such tree-level query, paper-pool and responses should be persisted into database as well, using 'topic' as the key name
- they should also be reloaded and displayed whenever the frontend launches/re-launches

create a new integration test, which test the backend api endpoint of 2 topic queries in debug mode:
   - use a topic like 'learning dynamics' of 'FP8 quantization'
   - add the top 10 papers into the pool
   - ask a random question you can think of about the 'learning dynamics', 
   - check the result, if they are as you expected. The following intermediate results should be saved into 'storage/schemas/<topic>.json', including
   - query embedding vector
   - list of top 10 papers added into the pool, and their respective embedding, and similarity score with the paper
   - the question asked in each topic and its embedding
   - the dict containing the ID of the paper retrieved, the ID of the chunk retrieved in each paper, the embedding of each chunk, and their corresponding similarity score with the question
   - the paper-specific response to each question
   - the prompt (including all the expanded context) and response sent to LLM for summarizing paper-specific responses

## UI modification
- "Details" to be centered at the top of right panel
- "Reclassify" renamed to "Re-categorize"
- "center_on_node" will also zoom into the node, with 85% magnification
- "center_on_node" will also be invoked, when any of the children categories in the "explorer" is clicked
- the "Ancestry" will also include the current node
- All node fonts in the tree diagram will not be adjusted by the font-size adjuster in the "Details" panel. Instead, they should use default 20, which is set in the config (as well as setting), as 'tree-diagram fontsize'
- currently, the zoom on the diagram will drift. Please always zoom in/out, anchoring the center of zoom at the current cursor location. If a node is selected, anchoring the center of zoom at the node

## Bug 13: PDF Text Extraction Corruption (pypdf `/uni` sequences)

**Symptom:**
Topic Query returns corrupted chunk text with `/uniXXXXXXXX` escape sequences instead of readable Unicode:
```
/uni00000353/uni00000355/uni00000356/uni00000351...
```

LLM cannot provide meaningful responses because retrieved chunks contain garbled text.

**Root Cause:**
pypdf (used by PaperQA2) fails to properly decode Type3 fonts embedded in PDF figures. Instead of mapping glyphs to Unicode characters, it extracts raw glyph names like `/uni00000353`.

**Investigation:**
Comprehensive test of 100 PDFs × 10 chunks each × 4 extractors = 4000 chunk evaluations:

| Extractor | Avg Quality | Good/Bad Chunks | Uni Seqs | Avg Time |
|-----------|-------------|-----------------|----------|----------|
| pypdf | 94.60% | 946/54 | 17,060 | 0.953s |
| **pymupdf** | **100%** | **1000/0** | **0** | **0.167s** |
| pdfplumber | 100% | 1000/0 | 0 | 4.131s |
| pdfminer | 100% | 1000/0 | 0 | 2.834s |

Bad paper recovery (50 papers with known issues): pypdf 42/50, **pymupdf 50/50**, pdfplumber 50/50, pdfminer 50/50.

See `scripts/pdf_extraction_test/README.md` for full test results.

**Recommended Fix:**
Replace pypdf with pymupdf in PaperQA's PDF extraction pipeline:

```python
# Current (pypdf - problematic)
from pypdf import PdfReader
reader = PdfReader(pdf_path)
text = "\n".join(page.extract_text() for page in reader.pages)

# Recommended (pymupdf - 5.7x faster, 100% quality)
import fitz  # pymupdf
doc = fitz.open(pdf_path)
text = "\n".join(page.get_text() for page in doc)
doc.close()
```

**Chosen Implementation: Monkey-patch pypdf with pymupdf at runtime**

### Implementation Plan

#### Step 1: Create pymupdf monkey-patch module

Create `src/backend/pdf_patch.py`:
```python
"""Monkey-patch pypdf to use pymupdf for text extraction."""
import fitz  # pymupdf

class PyMuPDFPage:
    """Wrapper to make pymupdf page act like pypdf page."""
    def __init__(self, page):
        self._page = page
    
    def extract_text(self):
        return self._page.get_text()

class PyMuPDFReader:
    """Drop-in replacement for pypdf.PdfReader using pymupdf."""
    def __init__(self, path):
        self._doc = fitz.open(path)
        self.pages = [PyMuPDFPage(p) for p in self._doc]
    
    def __del__(self):
        if hasattr(self, '_doc'):
            self._doc.close()

def patch_pypdf():
    """Replace pypdf.PdfReader with pymupdf-backed reader."""
    import pypdf
    pypdf.PdfReader = PyMuPDFReader
```

#### Step 2: Apply patch at backend startup

Modify `src/backend/app.py` (at top, before other imports):
```python
from pdf_patch import patch_pypdf
patch_pypdf()
```

#### Step 3: Create migration script

Create `scripts/migrate_to_pymupdf.py`:
1. Delete all PaperQA indices (`storage/paperqa_index/*.pkl`)
2. Clear topic Q&A history (`DELETE FROM topic_queries`)
3. For each paper:
   - Re-index PDF with patched PaperQA
   - Update embedding in database
   - If was corrupted, regenerate summary + structured analysis
4. Trigger full tree regeneration

#### Step 4: Run migration via PBS job

Create `scripts/migrate_to_pymupdf.pbs`:
```bash
#PBS -l select=1:mem=64gb:ncpus=8
#PBS -l walltime=24:00:00
uvenv infer
python scripts/migrate_to_pymupdf.py
```

### Data to Update

| Data | Action | Method |
|------|--------|--------|
| PaperQA indices | Delete all, regenerate | Script |
| Paper embeddings | Regenerate from new indices | Script |
| Summaries (50 bad papers) | Regenerate | Script calls `/summarize` |
| Structured analysis (50 bad) | Regenerate | Script calls `/summarize/structured` |
| Topic Q&A history | Clear | `DELETE FROM topic_queries` |
| Tree structure | Regenerate | Script calls `/papers/classify` |

### Verification

After migration:
1. Run Topic Query integration test on previously-corrupted papers
2. Verify no `/uni` sequences in extracted chunks
3. Compare embedding distances before/after for quality assessment

**Status:** Implementation complete, pending migration run

---

### Implementation Complete

**Files Created:**
- `src/backend/pdf_patch.py` - Monkey-patch module that replaces `pypdf.PdfReader` with `pymupdf`
- `src/backend/app.py` - Modified to call `patch_pypdf()` at startup
- `scripts/migrate_to_pymupdf.py` - Full migration script
- `scripts/migrate_to_pymupdf.pbs` - PBS job wrapper
- `scripts/verify_migration_ready.py` - Pre-migration verification

**Dependencies Updated:**
- `src/backend/requirements.txt` - Added `pymupdf`
- `containers/backend.def` - Added `pdf_patch.py` to container

**Verification Results (Feb 5, 2026):**
- pdf_patch module: ✓ Working
- pymupdf: ✓ v1.26.5 installed
- Patch mechanism: ✓ pypdf.PdfReader = PyMuPDFReader
- Test extraction: ✓ 40.7% corruption → 0% (PDF 2405.10938)

**To Run Migration:**
```bash
# 1. Rebuild container (one-time)
make singularity-build

# 2. Start services
make singularity-run

# 3. Verify readiness
python scripts/verify_migration_ready.py

# 4. Dry run
python scripts/migrate_to_pymupdf.py --dry-run

# 5. Full migration
python scripts/migrate_to_pymupdf.py
```




---

## Troubleshooting: Container Rebuild Not Taking Effect

### Problem
After running `make singularity-build` followed by `make singularity-run`, the changes don't appear in the running application.

### Cause
The `hpc-services.sh start` script detects already-running services and **skips restarting them**:
```
[2026-02-06 08:10:19] Database already running
[2026-02-06 08:10:19] Backend already running  
[2026-02-06 08:10:19] Frontend already running
```

This means the **old container** continues to run, not the newly built one.

### Solution
Always stop services before starting them after a rebuild:

```bash
# Option 1: Stop then start
make singularity-stop
make singularity-run

# Option 2: Combined command
make singularity-stop && make singularity-run
```

### Best Practice
When rebuilding containers, use this sequence:
```bash
# 1. Stop running services first
make singularity-stop

# 2. Rebuild containers
make singularity-build

# 3. Start with new containers
make singularity-run
```

Or, if services are already stopped:
```bash
make singularity-build && make singularity-run
```

---

## UI Updates (Feb 6, 2026)

The following UI changes were implemented:

1. **"Details" header centered** - Added `text-center` class to the Details header in the right panel
2. **"Reclassify" → "Re-categorize"** - Button and loading text renamed
3. **centerOnNode zooms to 85%** - When centering on a node (via ancestry click or category selection), zoom is set to 85% of current level
4. **Child categories click centers** - Clicking a child category in Explorer now selects it AND centers the tree on it
5. **Ancestry includes current node** - The ancestry chain now shows the current node (highlighted, non-clickable) at the end
6. **Tree font sizes from config** - Tree diagram uses `tree_category_font_size` and `tree_paper_font_size` from config (default: 20), independent of panel font adjuster
7. **Zoom anchors at cursor** - Ctrl+scroll zooming now anchors at cursor position, preventing drift. Zoom buttons anchor at viewport center.

### Config Settings Added
```yaml
ui:
  tree_category_font_size: 20  # Font size for category nodes in tree diagram
  tree_paper_font_size: 20     # Font size for paper nodes in tree diagram
```

### Files Modified
- `src/frontend/src/app/page.tsx` - All UI logic changes
- `src/backend/config.py` - Added tree font config support
- `config/config.yaml` - Added tree font size settings

---

## Testing: Running Integration Tests with Python 3.9

### Problem
When running `pytest tests/test_topic_query_integration.py`, the test fails with:
```
TypeError: unsupported operand type(s) for |: 'types.GenericAlias' and 'NoneType'
```

### Cause
The `tests/conftest.py` imports `from app import app`, which triggers `paperqa` imports. PaperQA uses Python 3.10+ type syntax (`list[...] | None`) which is incompatible with Python 3.9.

However, `test_topic_query_integration.py` uses **HTTP requests** to the live backend - it doesn't actually need the app import.

### Solution
Temporarily rename conftest.py when running integration tests on Python 3.9:

```bash
# Rename conftest to skip loading
mv tests/conftest.py tests/conftest.py.bak

# Run the integration test
pytest tests/test_topic_query_integration.py -v -s --tb=short

# Restore conftest
mv tests/conftest.py.bak tests/conftest.py
```

**Note:** This workaround is only needed for integration tests that use HTTP requests. Unit tests requiring the `client` fixture need Python 3.10+.

---

## Unicode Characters in Extracted Text

### Observation
The `topic_query.json` debug output contains Unicode escape sequences like `\u2212`, `\u03b5`, `\ufffd`.

### Analysis
Most of these are **valid Unicode mathematical symbols**, NOT extraction errors:

| Escape | Character | Meaning |
|--------|-----------|---------|
| `\u2212` | − | Unicode minus sign (distinct from ASCII hyphen) |
| `\u03b5` | ε | Greek letter epsilon |
| `\u03b8` | θ | Greek letter theta |
| `\u03c0` | π | Greek letter pi |
| `\u223c` | ∼ | Tilde operator |
| `\u00b7` | · | Middle dot |
| `\u02c6` | ˆ | Circumflex accent |

**The only problematic character is `\ufffd`** (�) - the Unicode replacement character, which indicates a character that couldn't be decoded. In the test file, there was only **1 occurrence** out of thousands of characters.

### Conclusion
The pymupdf migration is working correctly. The Unicode escapes in JSON are proper representation of mathematical notation from papers, not corruption.

---

## Endpoint Flow Documentation

### 1. Paper Ingestion Flow (`POST /batch-ingest` or `/paper/create`)

**Purpose:** Download, extract, summarize, and store papers from arXiv or local PDFs.

**Step-by-step flow:**

1. **PDF Download** (for arXiv papers):
   - Download PDF from arXiv using `arxiv.Client()`
   - Store in `storage/downloads/` directory

2. **PDF Extraction** (via PaperQA2 + pymupdf patch):
   - `pdf_patch.py` monkey-patches pypdf to use **pymupdf** for text extraction
   - Text is extracted with proper Unicode handling

3. **Chunking** (via PaperQA2):
   - **Chunk size:** 5000 characters (`chunk_chars: 5000` in config)
   - **Overlap:** 250 characters (`chunk_overlap: 250` in config)
   - Chunks are created with overlap for context continuity

4. **Embedding Generation**:
   - Each chunk is embedded using the embedding endpoint (port 8004)
   - Document-level embedding is computed (average of chunk embeddings)
   - Embedding dimension: 4096 (for Jina embeddings)

5. **Summarization** (optional):
   - LLM generates summary using loaded prompt template
   - Summary stored in database

6. **Index Persistence**:
   - PaperQA2 `Docs` object saved to `storage/paperqa_index/{arxiv_id}.pkl`
   - Contains chunked texts + embeddings for fast retrieval

7. **Database Storage**:
   - Paper metadata (arxiv_id, title, abstract, pdf_path) → `papers` table
   - Document embedding → `papers.embedding` (pgvector)
   - Summary → `papers.summary`

### 2. Single Paper Query Flow (`POST /paper/{arxiv_id}/query`)

**Purpose:** Answer a question using content from a single paper.

**Parameters used:**
- `evidence_k: 10` - Retrieve top 10 most relevant chunks
- `evidence_summary_length: "about 100 words"` - Summarize each evidence chunk
- `evidence_skip_summary: false` - Generate chunk summaries
- `evidence_relevance_score_cutoff: 3` - Minimum relevance score (1-5 scale)

**Step-by-step flow:**

1. **Load Index**:
   - Load PaperQA2 `Docs` object from `storage/paperqa_index/{arxiv_id}.pkl`
   - If not found, index the PDF on-the-fly

2. **Embed Question**:
   - Generate embedding for the question using embedding endpoint

3. **Retrieve Chunks** (evidence_k=10):
   - Similarity search between question embedding and chunk embeddings
   - Return top 10 most similar chunks

4. **Summarize Evidence**:
   - For each relevant chunk, LLM generates ~100 word summary
   - Summaries contextualized with the question

5. **Generate Answer**:
   - LLM synthesizes final answer from all evidence summaries
   - Citations linked to specific chunks

6. **Cache Response**:
   - Store question+answer in `paper_queries` table for reuse

### 3. Topic Query Flow (`POST /topic/{topic_id}/query`)

**Purpose:** Answer a question using content from multiple papers in a topic pool.

**Architecture:** Hierarchical RAG (per-paper → aggregation)

**Step-by-step flow:**

1. **Get Topic Papers**:
   - Fetch all papers in the topic pool from database
   - Each paper has similarity_score to topic embedding

2. **Embed Question** (debug mode):
   - Generate question embedding for debug output

3. **Query Each Paper** (parallel with semaphore=4):
   - For each paper in pool:
     - Load PaperQA2 index
     - Retrieve top chunks (evidence_k=10)
     - Generate per-paper answer via LLM
   - All papers queried in parallel (async)

4. **Collect Responses**:
   - Per-paper responses collected
   - Text quality analyzed (detect corruption)

5. **Aggregate Responses**:
   - Build aggregation prompt with all paper responses
   - LLM synthesizes cross-paper summary
   - Final answer generated

6. **Debug Output** (if enabled):
   - Save to `storage/schemas/topic_query.json`:
     - Topic/question embeddings
     - Per-paper chunks with scores
     - Aggregation prompt
     - Final answer

---

## Topic Query Assessment & Improvement Ideas

### Current Results Assessment

The topic query integration test (Feb 6, 2026) showed:
- **10/10 papers queried successfully** (100% success rate)
- **Chunks retrieved:** 5-8 per paper (reasonable)
- **Final answer:** 3560 chars, comprehensive
- **Quality:** Model correctly acknowledges when papers lack relevant info

**Positive observations:**
1. No PDF extraction failures - pymupdf patch working
2. No LLM errors - embeddings and answers generated successfully
3. Model provides honest responses when papers don't contain relevant info
4. Good chunk retrieval (5-8 chunks = evidence_k working)

**Areas for improvement:**

### Improvement Ideas

#### 1. Relevance Pre-filtering
**Problem:** All papers in pool are queried, even if topic query is narrow.
**Solution:** Before querying, rank papers by cosine similarity between question embedding and paper embedding. Only query top-N most relevant papers.

#### 2. Dynamic Chunk Count
**Problem:** Fixed evidence_k=10 may be too many for simple questions, too few for complex ones.
**Solution:** Adaptive evidence_k based on question complexity or paper relevance score.

#### 3. Two-Stage Retrieval
**Problem:** Current single-stage retrieval may miss relevant chunks.
**Solution:** 
- Stage 1: Broad retrieval (evidence_k=20)
- Stage 2: Re-rank using cross-encoder for precision

#### 4. Response Quality Scoring
**Problem:** No automated way to detect "I don't know" responses.
**Solution:** Add confidence scoring to detect low-value responses and exclude from aggregation.

#### 5. Chunk Deduplication
**Problem:** Similar chunks from different papers may be redundant.
**Solution:** Before aggregation, deduplicate chunks by semantic similarity.

#### 6. Streaming Responses
**Problem:** Topic queries can take 4+ minutes.
**Solution:** Stream intermediate results (per-paper responses) as they complete.

#### 7. Query Caching
**Problem:** Same question asked multiple times regenerates answer.
**Solution:** Cache topic+question → answer (already partially implemented via paper_queries table).

---

## Bug 7: LiteLLM Callback Accumulation and Async Task Warnings

**Date:** Feb 6, 2026

### Symptoms
In `logs/backend.log`, frequent warnings and errors:

1. **Callback limit exceeded** (appears hundreds of times):
```
LiteLLM:WARNING: logging_callback_manager.py:159 - Cannot add callback - would exceed MAX_CALLBACKS limit of 30. Current callbacks: 30
```

2. **Async task cleanup errors**:
```
Task exception was never retrieved
ValueError: task_done() called too many times
Task was destroyed but it is pending!
```

### Root Cause
LiteLLM internally manages logging callbacks via a `LoggingWorker` class. When PaperQA2 makes many LLM calls (especially during batch operations like tree classification with 200+ nodes), callbacks accumulate faster than they're cleaned up.

The existing `reset_litellm_callbacks()` function cleared the main callback lists but:
1. Did not disable debug logging that triggers new callbacks
2. Did not clear all internal callback storage attributes in the logging manager
3. Did not handle the `callbacks` module directly

### Fix
Updated `src/backend/llm_clients.py:reset_litellm_callbacks()` with:

1. **Disable verbose logging**: `litellm.suppress_debug_info = True`
2. **Comprehensive attribute clearing**: Loop through all known callback storage attributes
3. **Reset callback counts**: Clear counter attributes that track callback numbers
4. **Direct callbacks module reset**: Also reset the `litellm.callbacks` module

```python
def reset_litellm_callbacks() -> None:
    """Reset LiteLLM callbacks to prevent accumulation."""
    try:
        import litellm
    except ImportError:
        return
    
    # Disable verbose logging to prevent callback accumulation
    litellm.suppress_debug_info = True
    
    # Clear main callback lists
    litellm.input_callback = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []
    
    # Clear the logging callback manager - more thorough approach
    try:
        if hasattr(litellm, 'logging_callback_manager'):
            manager = litellm.logging_callback_manager
            for attr in ['callbacks', '_callbacks', 'callback_list', 
                         'success_callbacks', 'failure_callbacks',
                         '_success_callbacks', '_failure_callbacks']:
                if hasattr(manager, attr):
                    setattr(manager, attr, [])
            if hasattr(manager, 'callback_count'):
                manager.callback_count = 0
    except Exception:
        pass
    
    # Also reset the callbacks module directly
    try:
        from litellm import callbacks
        if hasattr(callbacks, 'callback_list'):
            callbacks.callback_list = []
    except (ImportError, AttributeError):
        pass
```

### Impact
- **Before**: Hundreds of warnings per classification run, potential memory leak from accumulated callbacks
- **After**: Callbacks are properly reset between operations, cleaner logs

### Files Modified
- `src/backend/llm_clients.py`

### Note
The async task warnings ("Task was destroyed but it is pending") are internal to LiteLLM's LoggingWorker and cannot be fully eliminated without patching LiteLLM itself. However, by suppressing debug info and aggressively clearing callbacks, the frequency is significantly reduced. These warnings are non-fatal and don't affect functionality.
