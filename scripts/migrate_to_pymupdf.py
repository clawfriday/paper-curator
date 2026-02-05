#!/usr/bin/env python3
"""
Migration script to re-index all PDFs with pymupdf backend.

Uses async IO for efficient parallel processing with controlled concurrency.
Tracks failed papers for later removal.

Usage:
    python scripts/migrate_to_pymupdf.py [--dry-run] [--skip-summaries] [--skip-tree]
    python scripts/migrate_to_pymupdf.py --concurrency 4  # 4 concurrent requests
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

# Apply pymupdf patch BEFORE any other imports
from pdf_patch import patch_pypdf, is_patched
patch_pypdf()

import aiohttp
import psycopg2
from psycopg2.extras import RealDictCursor


# Configuration
BACKEND_URL = "http://localhost:3100"
INDEX_DIR = Path(__file__).parent.parent / "storage" / "paperqa_index"
BAD_PAPERS_CONFIG = Path(__file__).parent / "pdf_extraction_test" / "test_pdf_config.json"
FAILED_PAPERS_FILE = Path(__file__).parent.parent / "logs" / "migration_failed_papers.json"

# Counters and tracking
success_count = 0
fail_count = 0
failed_papers: list[dict] = []  # Track failed papers for later removal


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


def save_failed_papers():
    """Save failed papers to JSON file."""
    global failed_papers
    FAILED_PAPERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILED_PAPERS_FILE, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "count": len(failed_papers),
            "papers": failed_papers
        }, f, indent=2)
    print(f"Failed papers saved to: {FAILED_PAPERS_FILE}")


async def reindex_paper_async(
    session: aiohttp.ClientSession,
    paper: dict,
    index: int,
    bad_arxiv_ids: set,
    skip_summaries: bool,
    timeout: int = 180
) -> tuple[bool, str | None]:
    """Re-index a single paper via API (async)."""
    global success_count, fail_count, failed_papers
    
    arxiv_id = paper["arxiv_id"]
    pdf_path = paper["pdf_path"]
    
    if not pdf_path or not Path(pdf_path).exists():
        fail_count += 1
        failed_papers.append({
            "id": paper["id"],
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "error": "PDF not found",
            "index": index
        })
        return False, f"[{index}] Skipping {arxiv_id}: PDF not found"
    
    try:
        async with session.post(
            f"{BACKEND_URL}/summarize",
            json={
                "pdf_path": pdf_path,
                "arxiv_id": arxiv_id,
                "text": None,
            },
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status == 200:
                success_count += 1
                if not skip_summaries and arxiv_id in bad_arxiv_ids:
                    return True, f"[{index}] ✓ {arxiv_id} (was corrupted)"
                return True, None
            else:
                fail_count += 1
                error_text = await response.text()
                failed_papers.append({
                    "id": paper["id"],
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title", ""),
                    "error": f"HTTP {response.status}",
                    "index": index
                })
                return False, f"[{index}] ✗ {arxiv_id}: HTTP {response.status}"
    except asyncio.TimeoutError:
        fail_count += 1
        failed_papers.append({
            "id": paper["id"],
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "error": "Timeout",
            "index": index
        })
        return False, f"[{index}] ✗ {arxiv_id}: Timeout"
    except aiohttp.ClientError as e:
        fail_count += 1
        failed_papers.append({
            "id": paper["id"],
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "error": str(type(e).__name__),
            "index": index
        })
        return False, f"[{index}] ✗ {arxiv_id}: {type(e).__name__}"
    except Exception as e:
        fail_count += 1
        failed_papers.append({
            "id": paper["id"],
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "error": str(e),
            "index": index
        })
        return False, f"[{index}] ✗ {arxiv_id}: {e}"


async def process_papers_async(
    papers: list[dict],
    start_from: int,
    bad_arxiv_ids: set,
    skip_summaries: bool,
    concurrency: int = 4,
) -> None:
    """Process papers with controlled concurrency using async IO."""
    global success_count, fail_count
    
    papers_to_process = papers[start_from:]
    total = len(papers_to_process)
    
    print(f"Total papers to process: {total}")
    print(f"Papers with known extraction issues: {len(bad_arxiv_ids)}")
    print(f"Concurrency: {concurrency}")
    print()
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create connector with connection pooling
    connector = aiohttp.TCPConnector(
        limit=concurrency * 2,
        limit_per_host=concurrency * 2,
        keepalive_timeout=30,
    )
    
    start_time = time.time()
    completed = 0
    
    async def process_with_semaphore(paper: dict, idx: int) -> tuple[bool, str | None]:
        async with semaphore:
            return await reindex_paper_async(
                session,
                paper,
                start_from + idx,
                bad_arxiv_ids,
                skip_summaries,
            )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create all tasks
        tasks = [
            asyncio.create_task(process_with_semaphore(paper, i))
            for i, paper in enumerate(papers_to_process)
        ]
        
        # Process results as they complete
        for coro in asyncio.as_completed(tasks):
            success, message = await coro
            completed += 1
            
            # Print messages for failures and corrupted papers
            if message:
                print(message, flush=True)
            
            # Print progress every 50 papers
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"\nProgress: {completed}/{total} papers "
                      f"({rate:.2f}/sec, ETA: {eta/60:.1f} min) "
                      f"[✓{success_count} ✗{fail_count}]", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\nRe-indexing complete in {elapsed/60:.1f} minutes")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Rate: {total/elapsed:.2f} papers/sec")
    
    # Save failed papers
    if failed_papers:
        save_failed_papers()


async def trigger_tree_regeneration_async(timeout: int = 1800) -> bool:
    """Trigger full tree regeneration via classify endpoint (async)."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BACKEND_URL}/papers/classify",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                return response.status == 200
    except Exception as e:
        print(f"Error triggering tree regeneration: {e}")
        return False


async def main_async():
    global success_count, fail_count, failed_papers
    
    parser = argparse.ArgumentParser(description="Migrate to pymupdf backend")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--skip-summaries", action="store_true", help="Skip regenerating summaries")
    parser.add_argument("--skip-tree", action="store_true", help="Skip tree regeneration")
    parser.add_argument("--start-from", type=int, default=0, help="Start from paper index")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests (default: 4)")
    parser.add_argument("--skip-delete", action="store_true", help="Skip deleting existing indices")
    args = parser.parse_args()
    
    print("=" * 60)
    print("pymupdf Migration Script (Async IO)")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Patch status: {'ACTIVE' if is_patched() else 'NOT ACTIVE'}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Check backend is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BACKEND_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"ERROR: Backend not healthy at {BACKEND_URL}")
                    sys.exit(1)
        print(f"Backend healthy at {BACKEND_URL}")
    except Exception as e:
        print(f"ERROR: Cannot connect to backend at {BACKEND_URL}: {e}")
        sys.exit(1)
    
    # Step 1: Delete all indices
    if not args.skip_delete:
        print("\n" + "-" * 40)
        print("Step 1: Delete existing PaperQA indices")
        print("-" * 40)
        deleted_indices = delete_all_indices(args.dry_run)
    else:
        print("\n[Skipping index deletion]")
        deleted_indices = 0
    
    # Step 2: Clear topic Q&A history
    print("\n" + "-" * 40)
    print("Step 2: Clear topic Q&A history")
    print("-" * 40)
    cleared_qa = clear_topic_qa_history(args.dry_run)
    
    # Step 3: Get all papers and re-index with async
    print("\n" + "-" * 40)
    print("Step 3: Re-index all papers (Async IO)")
    print("-" * 40)
    papers = get_all_papers()
    bad_arxiv_ids = get_bad_paper_arxiv_ids()
    
    if args.dry_run:
        print(f"[DRY RUN] Would re-index {len(papers) - args.start_from} papers")
    else:
        await process_papers_async(
            papers,
            args.start_from,
            bad_arxiv_ids,
            args.skip_summaries,
            args.concurrency,
        )
    
    # Step 4: Regenerate tree
    if not args.skip_tree:
        print("\n" + "-" * 40)
        print("Step 4: Regenerate tree structure")
        print("-" * 40)
        
        if args.dry_run:
            print("[DRY RUN] Would regenerate tree structure")
        else:
            print("Triggering tree regeneration (this may take 10-15 minutes)...")
            success = await trigger_tree_regeneration_async()
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
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    if failed_papers:
        print(f"  Failed papers saved to: {FAILED_PAPERS_FILE}")
    if not args.skip_tree:
        print("Tree regeneration: Triggered")
    print("=" * 60)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
