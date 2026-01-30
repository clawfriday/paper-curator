-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Papers table: stores all ingested papers
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL,
    abstract TEXT,
    summary TEXT,
    pdf_path TEXT,
    latex_path TEXT,
    pdf_url TEXT,
    published_at TIMESTAMPTZ,
    embedding vector(4096),  -- Qwen3-Embedding dimension
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tree state table: stores the taxonomy tree structure as JSONB
CREATE TABLE IF NOT EXISTS tree_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    tree_data JSONB NOT NULL DEFAULT '{"name": "AI Papers", "children": []}'::jsonb,
    node_names JSONB NOT NULL DEFAULT '{}'::jsonb,  -- Mapping of cluster_id -> name
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT single_tree CHECK (id = 1)  -- Only allow one row
);

-- Create index for JSONB queries
CREATE INDEX IF NOT EXISTS idx_tree_state_gin ON tree_state USING GIN (tree_data);

-- Insert initial empty tree
INSERT INTO tree_state (id, tree_data, node_names)
VALUES (1, '{"name": "AI Papers", "children": []}'::jsonb, '{}'::jsonb)
ON CONFLICT (id) DO NOTHING;

-- GitHub repos cache: stores repo lookup results
CREATE TABLE IF NOT EXISTS repo_cache (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    source VARCHAR(50) NOT NULL,  -- 'paperswithcode' or 'github'
    repo_url TEXT,
    repo_name TEXT,
    stars INTEGER,
    is_official BOOLEAN DEFAULT FALSE,
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_repo_cache_paper ON repo_cache(paper_id);

-- References table: stores extracted references from papers
CREATE TABLE IF NOT EXISTS paper_references (
    id SERIAL PRIMARY KEY,
    source_paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    cited_arxiv_id VARCHAR(50),  -- If we can resolve to arXiv
    cited_title TEXT NOT NULL,
    cited_authors TEXT[],
    cited_year INTEGER,
    citation_context TEXT,  -- The sentence/paragraph where it's cited
    explanation TEXT,  -- LLM-generated explanation (cached)
    explained_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_references_source ON paper_references(source_paper_id);
CREATE INDEX IF NOT EXISTS idx_references_arxiv ON paper_references(cited_arxiv_id);

-- Similar papers cache: stores similarity search results
CREATE TABLE IF NOT EXISTS similar_papers_cache (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    similar_arxiv_id VARCHAR(50),
    similar_title TEXT NOT NULL,
    similarity_score FLOAT,
    description TEXT,  -- Why it's similar
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_similar_paper ON similar_papers_cache(paper_id);


-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;
CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_tree_state_updated_at ON tree_state;
CREATE TRIGGER update_tree_state_updated_at
    BEFORE UPDATE ON tree_state
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
