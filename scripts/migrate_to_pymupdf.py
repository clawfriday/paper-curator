#!/usr/bin/env python3
"""
Migration script to re-index all PDFs with pymupdf backend.

This script:
1. Deletes all existing PaperQA indices
2. Clears topic Q&A history
3. Re-indexes all papers with pymupdf-backed extraction
4. Updates paper embeddings in database
5. Regenerates summaries for corrupted papers
6. Triggers full tree regeneration

Usage:
    python scripts/migrate_to_pymupdf.py [--dry-run] [--skip-summaries] [--skip-tree]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

# Apply pymupdf patch BEFORE any other imports
from pdf_patch import patch_pypdf, is_patched
patch_pypdf()

import requests
import psycopg2
from psycopg2.extras import RealDictCursor


# Configuration
BACKEND_URL = "http://localhost:3100"
INDEX_DIR = Path(__file__).parent.parent / "storage" / "paperqa_index"
BAD_PAPERS_CONFIG = Path(__file__).parent / "pdf_extraction_test" / "test_pdf_config.json"


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        database=os.environ.get("PGDATABASE", "paper_curator"),
        user=os.environ.get("PGUSER", "curator"),
        password=os.environ.get("PGPASSWORD", "curator123"),
    )


def delete_all_indices(dry_run: bool = False) -> int:
    """Delete all PaperQA index files."""
    if not INDEX_DIR.exists():
        print(f"Index directory not found: {INDEX_DIR}")
        return 0
    
    pkl_files = list(INDEX_DIR.glob("*.pkl"))
    print(f"Found {len(pkl_files)} index files to delete")
    
    if dry_run:
        print("[DRY RUN] Would delete all index files")
        return len(pkl_files)
    
    for f in pkl_files:
        f.unlink()
    
    print(f"Deleted {len(pkl_files)} index files")
    return len(pkl_files)


def clear_topic_qa_history(dry_run: bool = False) -> int:
    """Clear all topic Q&A history from database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM topic_queries")
            count = cur.fetchone()[0]
            print(f"Found {count} topic Q&A entries to delete")
            
            if dry_run:
                print("[DRY RUN] Would delete all topic Q&A entries")
                return count
            
            cur.execute("DELETE FROM topic_queries")
            conn.commit()
            print(f"Deleted {count} topic Q&A entries")
            return count
    finally:
        conn.close()


def get_all_papers() -> list[dict]:
    """Get all papers from database."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, arxiv_id, title, pdf_path
                FROM papers
                WHERE pdf_path IS NOT NULL
                ORDER BY id
            """)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_bad_paper_arxiv_ids() -> set[str]:
    """Get list of arxiv IDs that had extraction issues."""
    if not BAD_PAPERS_CONFIG.exists():
        print(f"Warning: Bad papers config not found: {BAD_PAPERS_CONFIG}")
        return set()
    
    with open(BAD_PAPERS_CONFIG) as f:
        config = json.load(f)
    
    return {p["arxiv_id"] for p in config.get("bad_papers", [])}


def reindex_paper(arxiv_id: str, pdf_path: str, timeout: int = 120) -> bool:
    """Re-index a single paper via API."""
    try:
        # Use the summarize endpoint which also indexes
        response = requests.post(
            f"{BACKEND_URL}/summarize",
            json={
                "pdf_path": pdf_path,
                "arxiv_id": arxiv_id,
                "text": None,  # Force PDF extraction
            },
            timeout=timeout,
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"  Error re-indexing {arxiv_id}: {e}")
        return False


def regenerate_summary(arxiv_id: str, pdf_path: str, timeout: int = 180) -> bool:
    """Regenerate summary for a paper."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/summarize",
            json={
                "pdf_path": pdf_path,
                "arxiv_id": arxiv_id,
            },
            timeout=timeout,
        )
        if response.status_code == 200:
            result = response.json()
            # Update summary in database
            conn = get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE papers SET summary = %s WHERE arxiv_id = %s",
                        (result.get("summary", ""), arxiv_id),
                    )
                    conn.commit()
            finally:
                conn.close()
            return True
        return False
    except requests.exceptions.RequestException as e:
        print(f"  Error regenerating summary for {arxiv_id}: {e}")
        return False


def regenerate_structured_analysis(arxiv_id: str, pdf_path: str, timeout: int = 300) -> bool:
    """Regenerate structured analysis for a paper."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/summarize/structured",
            json={
                "pdf_path": pdf_path,
                "arxiv_id": arxiv_id,
            },
            timeout=timeout,
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"  Error regenerating structured analysis for {arxiv_id}: {e}")
        return False


def trigger_tree_regeneration(timeout: int = 1800) -> bool:
    """Trigger full tree regeneration via classify endpoint."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/papers/classify",
            timeout=timeout,
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error triggering tree regeneration: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate to pymupdf backend")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--skip-summaries", action="store_true", help="Skip regenerating summaries")
    parser.add_argument("--skip-tree", action="store_true", help="Skip tree regeneration")
    parser.add_argument("--start-from", type=int, default=0, help="Start from paper index (for resuming)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("pymupdf Migration Script")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Patch status: {'ACTIVE' if is_patched() else 'NOT ACTIVE'}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Check backend is running
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"ERROR: Backend not healthy at {BACKEND_URL}")
            sys.exit(1)
        print(f"Backend healthy at {BACKEND_URL}")
    except requests.exceptions.RequestException:
        print(f"ERROR: Cannot connect to backend at {BACKEND_URL}")
        print("Please ensure the backend is running before migration")
        sys.exit(1)
    
    # Step 1: Delete all indices
    print("\n" + "-" * 40)
    print("Step 1: Delete existing PaperQA indices")
    print("-" * 40)
    deleted_indices = delete_all_indices(args.dry_run)
    
    # Step 2: Clear topic Q&A history
    print("\n" + "-" * 40)
    print("Step 2: Clear topic Q&A history")
    print("-" * 40)
    cleared_qa = clear_topic_qa_history(args.dry_run)
    
    # Step 3: Get all papers and re-index
    print("\n" + "-" * 40)
    print("Step 3: Re-index all papers")
    print("-" * 40)
    papers = get_all_papers()
    bad_arxiv_ids = get_bad_paper_arxiv_ids()
    print(f"Total papers to process: {len(papers)}")
    print(f"Papers with known extraction issues: {len(bad_arxiv_ids)}")
    
    if args.dry_run:
        print("[DRY RUN] Would re-index all papers")
    else:
        success_count = 0
        fail_count = 0
        
        for i, paper in enumerate(papers[args.start_from:], start=args.start_from):
            if i % 50 == 0:
                print(f"\nProgress: {i}/{len(papers)} papers")
            
            arxiv_id = paper["arxiv_id"]
            pdf_path = paper["pdf_path"]
            
            if not pdf_path or not Path(pdf_path).exists():
                print(f"  [{i}] Skipping {arxiv_id}: PDF not found")
                fail_count += 1
                continue
            
            success = reindex_paper(arxiv_id, pdf_path)
            if success:
                success_count += 1
                # Regenerate summary for bad papers
                if not args.skip_summaries and arxiv_id in bad_arxiv_ids:
                    print(f"  [{i}] Regenerating summary for {arxiv_id} (was corrupted)")
                    regenerate_summary(arxiv_id, pdf_path)
            else:
                fail_count += 1
                print(f"  [{i}] Failed to re-index {arxiv_id}")
        
        print(f"\nRe-indexing complete: {success_count} success, {fail_count} failed")
    
    # Step 4: Regenerate tree
    if not args.skip_tree:
        print("\n" + "-" * 40)
        print("Step 4: Regenerate tree structure")
        print("-" * 40)
        
        if args.dry_run:
            print("[DRY RUN] Would regenerate tree structure")
        else:
            print("Triggering tree regeneration (this may take 10-15 minutes)...")
            success = trigger_tree_regeneration()
            if success:
                print("Tree regeneration triggered successfully")
            else:
                print("WARNING: Tree regeneration failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"End time: {datetime.now().isoformat()}")
    print(f"Indices deleted: {deleted_indices}")
    print(f"Q&A entries cleared: {cleared_qa}")
    print(f"Papers processed: {len(papers) - args.start_from}")
    if not args.skip_tree:
        print("Tree regeneration: Triggered")
    print("=" * 60)


if __name__ == "__main__":
    main()
