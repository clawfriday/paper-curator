# Server & UI Restart Checklist

## ✅ Pre-Restart Checks

### 1. Database Schema
- [x] `tree_state` table defined in `init.sql` with JSONB storage
- [x] Uses `node_id` prefix consistently (e.g., `node_abc123`)
- [x] Indexes created for JSONB queries
- [x] Trigger for `updated_at` timestamp

**Action Required:** When you restart, the database will be initialized with `tree_state` table automatically via `init.sql` mounted in Docker.

### 2. Backend Code
- [x] `db.py` - Uses `tree_state` table, renamed `find_paper_cluster_id` to `find_paper_node_id`
- [x] `clustering.py` - Complete rewrite:
  - Uses `node_` prefix consistently
  - L2-normalizes embeddings before clustering
  - Uses sklearn's silhouette_score
  - BisectingKMeans fallback for robust splitting
  - Builds tree directly in frontend format
- [x] `app.py` - Updated to use new db function names
- [x] `naming.py` - Updated imports and documentation
- [x] All imports working correctly

### 3. Frontend Code
- [x] `page.tsx` - Expects `{name, children: [...]}` format
- [x] API route `/api/papers/classify` exists
- [x] Tree loading from `/api/tree` endpoint

### 4. Test Coverage
- [x] All papers have embeddings
- [x] Tree builds successfully
- [x] All papers present in tree
- [x] Branching factor constraint respected
- [x] No empty nodes
- [x] No circular references
- [x] All node IDs unique

## ⚠️ Important Notes

### Database State
- The `tree_state` table stores the tree as JSONB in a single row
- Node IDs now use `node_` prefix (e.g., `node_abc123`)
- Names are stored both in tree structure and `node_names` mapping

### First Run After Restart
1. **Tree will be empty** - You'll need to run `/papers/classify` to build the tree
2. **All papers need embeddings** - Classification requires embeddings for all papers
3. **Naming will run automatically** - After classification, naming runs level-by-level

## 🚀 Restart Steps

1. **Stop current services** (if running):
   ```bash
   docker compose -f src/compose.yml down
   ```

2. **Start services with rebuild**:
   ```bash
   docker compose -f src/compose.yml up --build
   ```

3. **Verify database**:
   ```bash
   docker compose -f src/compose.yml exec db psql -U curator -d paper_curator -c "\d tree_state"
   ```

4. **Build tree** (if needed):
   - Go to UI and click "Re-classify" or
   - Call `POST /api/papers/classify`

## ✅ Phase 4 Implementation Complete

All clustering and naming logic has been rewritten:
- Unified ID system (`node_` prefix)
- L2-normalized embeddings for cosine-similarity clustering
- BisectingKMeans fallback for robust splitting
- Direct frontend format building
- Comprehensive test coverage
