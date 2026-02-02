"""Database operations for paper-curator."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector


def get_connection_string() -> str:
    """Get database connection string from environment."""
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://curator:curator@localhost:5432/paper_curator"
    )


@contextmanager
def get_db() -> Generator[psycopg2.extensions.connection, None, None]:
    """Get a database connection with pgvector support."""
    conn = psycopg2.connect(get_connection_string())
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# Papers CRUD
# =============================================================================

def _ensure_structured_summary_column(conn: psycopg2.extensions.connection) -> None:
    """Ensure structured_summary column exists on papers table."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE papers ADD COLUMN IF NOT EXISTS structured_summary JSONB"
        )
        conn.commit()


def create_paper(
    arxiv_id: str,
    title: str,
    authors: list[str],
    abstract: Optional[str] = None,
    summary: Optional[str] = None,
    abbreviation: Optional[str] = None,
    pdf_path: Optional[str] = None,
    latex_path: Optional[str] = None,
    pdf_url: Optional[str] = None,
    published_at: Optional[str] = None,
    embedding: Optional[list[float]] = None,
) -> int:
    """Create a paper and return its ID."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO papers (arxiv_id, title, authors, abstract, summary, abbreviation,
                                    pdf_path, latex_path, pdf_url, published_at, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    abstract = EXCLUDED.abstract,
                    summary = COALESCE(EXCLUDED.summary, papers.summary),
                    abbreviation = COALESCE(EXCLUDED.abbreviation, papers.abbreviation),
                    pdf_path = COALESCE(EXCLUDED.pdf_path, papers.pdf_path),
                    latex_path = COALESCE(EXCLUDED.latex_path, papers.latex_path),
                    pdf_url = COALESCE(EXCLUDED.pdf_url, papers.pdf_url),
                    published_at = COALESCE(EXCLUDED.published_at, papers.published_at),
                    embedding = COALESCE(EXCLUDED.embedding, papers.embedding)
                RETURNING id
                """,
                (arxiv_id, title, authors, abstract, summary, abbreviation, pdf_path, 
                 latex_path, pdf_url, published_at, embedding)
            )
            paper_id = cur.fetchone()[0]
            conn.commit()
            return paper_id


def get_paper_by_arxiv_id(arxiv_id: str) -> Optional[dict[str, Any]]:
    """Get paper by arXiv ID."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM papers WHERE arxiv_id = %s", (arxiv_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_paper_by_id(paper_id: int) -> Optional[dict[str, Any]]:
    """Get paper by ID."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM papers WHERE id = %s", (paper_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_all_papers() -> list[dict[str, Any]]:
    """Get all papers."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM papers ORDER BY created_at DESC")
            return [dict(row) for row in cur.fetchall()]


def update_paper_embedding(paper_id: int, embedding: list[float]) -> None:
    """Update paper embedding."""
    import numpy as np
    embedding_arr = np.array(embedding, dtype=np.float32)
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE papers SET embedding = %s::vector WHERE id = %s",
                (embedding_arr, paper_id)
            )
            conn.commit()


def update_paper_summary(paper_id: int, summary: str) -> None:
    """Update paper summary."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE papers SET summary = %s WHERE id = %s",
                (summary, paper_id)
            )
            conn.commit()


def update_paper_abbreviation(paper_id: int, abbreviation: str) -> None:
    """Update paper abbreviation (short display name)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE papers SET abbreviation = %s WHERE id = %s",
                (abbreviation, paper_id)
            )
            conn.commit()


def get_paper_abbreviation(paper_id: int) -> Optional[str]:
    """Get paper abbreviation by paper ID."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT abbreviation FROM papers WHERE id = %s", (paper_id,))
            row = cur.fetchone()
            return row[0] if row and row[0] else None


def update_paper_structured_summary(paper_id: int, structured_summary: dict) -> None:
    """Update paper structured summary (detailed analysis)."""
    import json
    with get_db() as conn:
        _ensure_structured_summary_column(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE papers SET structured_summary = %s WHERE id = %s",
                (json.dumps(structured_summary), paper_id)
            )
            conn.commit()


def get_paper_structured_summary(paper_id: int) -> Optional[dict]:
    """Get paper structured summary."""
    import json
    with get_db() as conn:
        _ensure_structured_summary_column(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT structured_summary FROM papers WHERE id = %s",
                (paper_id,)
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])
            return None


def find_similar_papers(embedding: list[float], limit: int = 5, exclude_id: Optional[int] = None) -> list[dict[str, Any]]:
    """Find similar papers by embedding using cosine distance."""
    import numpy as np
    # Convert to numpy array for pgvector compatibility
    embedding_arr = np.array(embedding, dtype=np.float32)
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if exclude_id:
                cur.execute(
                    """
                    SELECT id, arxiv_id, title, authors, abstract, summary,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM papers
                    WHERE embedding IS NOT NULL AND id != %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_arr, exclude_id, embedding_arr, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT id, arxiv_id, title, authors, abstract, summary,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM papers
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_arr, embedding_arr, limit)
                )
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Tree CRUD
# =============================================================================

def get_tree() -> dict[str, Any]:
    """Get full tree structure from JSONB and enrich with paper metadata.
    
    Returns:
        Frontend format tree structure with paper metadata:
        {
            "name": "AI Papers",
            "children": [
                {
                    "name": "Category Name",
                    "node_id": "node_abc",
                    "node_type": "category",
                    "children": [...]
                },
                {
                    "name": "Paper Name",
                    "node_id": "node_def",
                    "node_type": "paper",
                    "paper_id": 123,
                    "attributes": {
                        "arxivId": "...",
                        "title": "...",
                        ...
                    }
                }
            ]
        }
    """
    import json
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT tree_data, node_names FROM tree_state WHERE id = 1")
            row = cur.fetchone()
            if not row or not row.get("tree_data"):
                return {"name": "AI Papers", "children": []}
            
            tree_data = row["tree_data"]
            tree_structure = tree_data if isinstance(tree_data, dict) else json.loads(tree_data)
            node_names_raw = row.get("node_names") or {}
            node_names = node_names_raw if isinstance(node_names_raw, dict) else json.loads(node_names_raw)
    
    # Enrich tree with paper metadata
    def enrich_node(node: dict[str, Any]) -> dict[str, Any]:
        """Recursively enrich nodes with paper metadata."""
        if node.get("node_type") == "paper" and node.get("paper_id"):
            # Fetch paper data from database
            paper = get_paper_by_id(node["paper_id"])
            if paper:
                node["attributes"] = {
                    "arxivId": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "authors": paper.get("authors") or [],
                    "summary": paper.get("summary"),
                    "pdfPath": paper.get("pdf_path"),
                }
                # Use stored abbreviation from database (preferred)
                db_abbrev = paper.get("abbreviation")
                if db_abbrev:
                    node["name"] = db_abbrev
                else:
                    # Fallback: use existing node name or generate from title
                    existing_name = (node.get("name") or "").strip()
                    if not existing_name or existing_name.startswith("paper_"):
                        title = paper.get("title", "")
                        if title:
                            # Extract abbreviation if present before colon
                            if ":" in title:
                                prefix = title.split(":")[0].strip()
                                if len(prefix) <= 20:
                                    node["name"] = prefix
                                else:
                                    node["name"] = title[:20] + ".."
                            else:
                                node["name"] = title[:20] + ".." if len(title) > 20 else title
        elif node.get("node_type") == "category":
            node_id = node.get("node_id")
            if node_id and node_names:
                named = (node_names.get(node_id) or "").strip()
                if named:
                    current = (node.get("name") or "").strip()
                    if not current or current == node_id or current.startswith("node_"):
                        node["name"] = named
        
        # Recursively enrich children
        if node.get("children"):
            node["children"] = [enrich_node(child) for child in node["children"]]
        
        return node
    
    return enrich_node(tree_structure)


def save_tree(tree_structure: dict[str, Any], node_names: dict[str, str] | None = None) -> None:
    """Save tree structure to database as JSONB.
    
    Args:
        tree_structure: Frontend format tree structure {name, children: [...]}
        node_names: Optional dictionary mapping node_id to node name
    """
    import json
    
    if node_names is None:
        node_names = {}
        def extract_names(node: dict[str, Any]):
            if node.get("node_id"):
                node_names[node["node_id"]] = node.get("name", "")
            if node.get("children"):
                for child in node["children"]:
                    extract_names(child)
        extract_names(tree_structure)
    
    with get_db() as conn:
        with conn.cursor() as cur:
            # Lock the row to prevent concurrent updates
            cur.execute("SELECT * FROM tree_state WHERE id = 1 FOR UPDATE")
            
            cur.execute(
                """
                UPDATE tree_state 
                SET tree_data = %s::jsonb, node_names = %s::jsonb, updated_at = NOW()
                WHERE id = 1
                """,
                (json.dumps(tree_structure), json.dumps(node_names))
            )
            conn.commit()


def get_tree_node_names() -> dict[str, str]:
    """Get node names mapping from database.
    
    Returns:
        Dictionary mapping node_id -> name
    """
    import json
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT node_names FROM tree_state WHERE id = 1")
            row = cur.fetchone()
            if not row or not row.get("node_names"):
                return {}
            
            node_names_data = row["node_names"]
            return node_names_data if isinstance(node_names_data, dict) else json.loads(node_names_data)


def find_paper_node_id(paper_id: int) -> str | None:
    """Find the node_id for a paper in the tree.
    
    Args:
        paper_id: Integer paper ID
        
    Returns:
        Node ID if found, None otherwise
    """
    tree = get_tree()
    
    def search_node(node: dict[str, Any]) -> str | None:
        if node.get("node_type") == "paper" and node.get("paper_id") == paper_id:
            return node.get("node_id")
        if node.get("children"):
            for child in node["children"]:
                result = search_node(child)
                if result:
                    return result
        return None
    
    return search_node(tree)


def update_tree_node_name(node_id: str, name: str) -> None:
    """Update a tree node's display name in JSONB structure.
    
    Args:
        node_id: Node ID to update
        name: New name for the node
    """
    import json
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Lock the row to prevent concurrent updates
            cur.execute("SELECT tree_data, node_names FROM tree_state WHERE id = 1 FOR UPDATE")
            row = cur.fetchone()
            if not row:
                return
            
            tree_data_raw = row["tree_data"]
            node_names_raw = row["node_names"]
            tree_data = tree_data_raw if isinstance(tree_data_raw, dict) else json.loads(tree_data_raw)
            node_names = node_names_raw if isinstance(node_names_raw, dict) else json.loads(node_names_raw)
            
            # Update name in tree structure
            def update_name_in_tree(node: dict[str, Any]) -> None:
                if node.get("node_id") == node_id:
                    node["name"] = name
                if node.get("children"):
                    for child in node["children"]:
                        update_name_in_tree(child)
            
            update_name_in_tree(tree_data)
            
            # Update node_names mapping
            node_names[node_id] = name
            
            # Save back
            cur.execute(
                """
                UPDATE tree_state 
                SET tree_data = %s::jsonb, node_names = %s::jsonb, updated_at = NOW()
                WHERE id = 1
                """,
                (json.dumps(tree_data), json.dumps(node_names))
            )
            conn.commit()


def delete_paper(paper_id: int) -> None:
    """Delete a paper and all its associated data.
    
    CASCADE constraints will automatically delete:
    - paper_references (source_paper_id)
    - similar_papers_cache (paper_id)
    - repo_cache (paper_id)
    - paper_queries (paper_id)
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM papers WHERE id = %s", (paper_id,))
            conn.commit()




def get_all_categories_with_counts() -> list[dict[str, Any]]:
    """Get all categories with their paper counts from tree structure."""
    tree = get_tree()
    categories = []
    
    def traverse(node: dict[str, Any]) -> None:
        if node.get("node_type") == "category":
            # Count papers in this category
            paper_count = 0
            def count_papers(n: dict[str, Any]) -> int:
                count = 0
                if n.get("node_type") == "paper":
                    return 1
                if n.get("children"):
                    for child in n["children"]:
                        count += count_papers(child)
                return count
            
            paper_count = count_papers(node)
            categories.append({
                "node_id": node.get("node_id"),
                "name": node.get("name"),
                "paper_count": paper_count,
            })
        
        if node.get("children"):
            for child in node["children"]:
                traverse(child)
    
    traverse(tree)
    return sorted(categories, key=lambda x: x["paper_count"], reverse=True)


def get_category_paper_count(category_name: str) -> int:
    """Get number of papers in a category by name."""
    tree = get_tree()
    
    def find_category(node: dict[str, Any]) -> dict[str, Any] | None:
        if node.get("node_type") == "category" and node.get("name") == category_name:
            return node
        if node.get("children"):
            for child in node["children"]:
                result = find_category(child)
                if result:
                    return result
        return None
    
    category = find_category(tree)
    if not category:
        return 0
    
    def count_papers(n: dict[str, Any]) -> int:
        count = 0
        if n.get("node_type") == "paper":
            return 1
        if n.get("children"):
            for child in n["children"]:
                count += count_papers(child)
        return count
    
    return count_papers(category)


def get_category_paper_count_by_id(category_node_id: str) -> int:
    """Get number of papers in a category by node_id."""
    tree = get_tree()
    
    def find_category(node: dict[str, Any]) -> dict[str, Any] | None:
        if node.get("node_id") == category_node_id:
            return node
        if node.get("children"):
            for child in node["children"]:
                result = find_category(child)
                if result:
                    return result
        return None
    
    category = find_category(tree)
    if not category:
        return 0
    
    def count_papers(n: dict[str, Any]) -> int:
        count = 0
        if n.get("node_type") == "paper":
            return 1
        if n.get("children"):
            for child in n["children"]:
                count += count_papers(child)
        return count
    
    return count_papers(category)


def get_papers_in_category(category_node_id: str) -> list[dict[str, Any]]:
    """Get all papers in a category with their details."""
    tree = get_tree()
    papers = []
    
    def find_category(node: dict[str, Any]) -> dict[str, Any] | None:
        if node.get("node_id") == category_node_id:
            return node
        if node.get("children"):
            for child in node["children"]:
                result = find_category(child)
                if result:
                    return result
        return None
    
    category = find_category(tree)
    if not category:
        return []
    
    def collect_papers(n: dict[str, Any]) -> None:
        if n.get("node_type") == "paper" and n.get("paper_id"):
            paper = get_paper_by_id(n["paper_id"])
            if paper:
                papers.append({
                    "node_id": n.get("node_id"),
                    "display_name": n.get("name"),
                    **paper,
                })
        if n.get("children"):
            for child in n["children"]:
                collect_papers(child)
    
    collect_papers(category)
    return papers



# =============================================================================
# Repo Cache
# =============================================================================

def cache_repo(
    paper_id: int,
    source: str,
    repo_url: Optional[str],
    repo_name: Optional[str] = None,
    stars: Optional[int] = None,
    is_official: bool = False,
) -> None:
    """Cache a repo lookup result."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO repo_cache (paper_id, source, repo_url, repo_name, stars, is_official)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (paper_id, source, repo_url, repo_name, stars, is_official)
            )
            conn.commit()


def get_cached_repos(paper_id: int) -> list[dict[str, Any]]:
    """Get cached repos for a paper."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM repo_cache WHERE paper_id = %s ORDER BY is_official DESC, stars DESC NULLS LAST",
                (paper_id,)
            )
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# References Cache
# =============================================================================

def add_reference(
    source_paper_id: int,
    cited_title: str,
    cited_arxiv_id: Optional[str] = None,
    cited_authors: Optional[list[str]] = None,
    cited_year: Optional[int] = None,
    citation_context: Optional[str] = None,
) -> int:
    """Add a reference and return its ID."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_references (source_paper_id, cited_title, cited_arxiv_id, 
                                              cited_authors, cited_year, citation_context)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (source_paper_id, cited_title, cited_arxiv_id, cited_authors, cited_year, citation_context)
            )
            ref_id = cur.fetchone()[0]
            conn.commit()
            return ref_id


def get_references(paper_id: int) -> list[dict[str, Any]]:
    """Get references for a paper."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM paper_references WHERE source_paper_id = %s ORDER BY id",
                (paper_id,)
            )
            return [dict(row) for row in cur.fetchall()]


def update_reference_explanation(ref_id: int, explanation: str) -> None:
    """Update reference explanation (cache LLM result)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE paper_references SET explanation = %s, explained_at = NOW() WHERE id = %s",
                (explanation, ref_id)
            )
            conn.commit()


# =============================================================================
# Similar Papers Cache
# =============================================================================

def cache_similar_paper(
    paper_id: int,
    similar_arxiv_id: Optional[str],
    similar_title: str,
    similarity_score: Optional[float] = None,
    description: Optional[str] = None,
) -> None:
    """Cache a similar paper result."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO similar_papers_cache (paper_id, similar_arxiv_id, similar_title, similarity_score, description)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (paper_id, similar_arxiv_id, similar_title, similarity_score, description)
            )
            conn.commit()


def get_cached_similar_papers(paper_id: int) -> list[dict[str, Any]]:
    """Get cached similar papers with field names matching frontend expectations."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT 
                    similar_arxiv_id as arxiv_id,
                    similar_title as title,
                    similarity_score as similarity,
                    description
                FROM similar_papers_cache 
                WHERE paper_id = %s 
                ORDER BY similarity_score DESC NULLS LAST
                """,
                (paper_id,)
            )
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Query History
# =============================================================================

def add_query(
    paper_id: int,
    question: str,
    answer: str,
    model: Optional[str] = None,
) -> int:
    """Save a QA query and return its ID."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_queries (paper_id, question, answer, model)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (paper_id, question, answer, model)
            )
            query_id = cur.fetchone()[0]
            conn.commit()
            return query_id


def get_queries(paper_id: int) -> list[dict[str, Any]]:
    """Get query history for a paper."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM paper_queries WHERE paper_id = %s ORDER BY created_at DESC",
                (paper_id,)
            )
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Get All Cached Data for a Paper
# =============================================================================

def get_paper_cached_data(paper_id: int) -> dict[str, Any]:
    """Get all cached data for a paper (repos, refs, similar, queries)."""
    return {
        "repos": get_cached_repos(paper_id),
        "references": get_references(paper_id),
        "similar_papers": get_cached_similar_papers(paper_id),
        "queries": get_queries(paper_id),
    }
