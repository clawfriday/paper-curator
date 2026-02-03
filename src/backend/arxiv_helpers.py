"""arXiv helper utilities."""
from __future__ import annotations

import re
import time
from typing import Optional

import arxiv
from fastapi import HTTPException


def require_identifier(arxiv_id: Optional[str], url: Optional[str]) -> str:
    """Extract arXiv ID from provided arxiv_id or URL."""
    if arxiv_id:
        return arxiv_id
    if url:
        match = re.search(r"arxiv.org/(abs|pdf)/([0-9.]+)", url)
        if match:
            return match.group(2)
    raise HTTPException(status_code=400, detail="Provide arxiv_id or url.")


def extract_arxiv_ids_from_text(text: str) -> list[str]:
    """Extract arXiv IDs from text (handles URLs and bare IDs)."""
    arxiv_ids = set()
    # URL patterns
    url_pattern = r"arxiv\\.org/(abs|pdf)/([0-9.]+)"
    for match in re.finditer(url_pattern, text):
        arxiv_id = match.group(2)
        arxiv_id = arxiv_id.replace(".pdf", "")
        arxiv_ids.add(arxiv_id)
    # Bare ID patterns
    bare_pattern = r"\\b([0-9]{4}\\.[0-9]{4,5})(v\\d+)?\\b"
    for match in re.finditer(bare_pattern, text):
        arxiv_ids.add(match.group(1))
    return list(arxiv_ids)


def download_arxiv_pdf(identifier: str, output_dir: str, max_retries: int = 3) -> str:
    """Download arXiv PDF with retry logic."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[identifier])
    results = list(client.results(search))
    if not results:
        raise HTTPException(status_code=404, detail="No arXiv result found.")
    result = results[0]

    for attempt in range(max_retries):
        try:
            return result.download_pdf(dirpath=output_dir)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)


async def download_arxiv_pdf_async(arxiv_id: str, max_retries: int = 3) -> str:
    """Download arXiv PDF into storage/downloads with retry logic."""
    import asyncio
    import pathlib

    output_dir = pathlib.Path("storage/downloads")
    output_dir.mkdir(parents=True, exist_ok=True)
    last_error = None

    for attempt in range(max_retries):
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(client.results(search))
            if not results:
                raise HTTPException(status_code=404, detail=f"arXiv paper not found: {arxiv_id}")
            result = results[0]
            pdf_path = result.download_pdf(dirpath=str(output_dir))
            return str(pdf_path)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            error_msg = str(e)
            if "retrieval incomplete" in error_msg or "urlopen error" in error_msg:
                raise Exception(
                    f"PDF download failed after {max_retries} attempts: {error_msg}. "
                    "The PDF may be too large or the connection was interrupted."
                )
            raise Exception(f"PDF download failed after {max_retries} attempts: {error_msg}")

    raise Exception(f"PDF download failed: {last_error}")
