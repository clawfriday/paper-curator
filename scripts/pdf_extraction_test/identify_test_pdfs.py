#!/usr/bin/env python3
"""
Identify 50 bad and 50 good quality PDFs based on text extraction quality.
Uses pypdf to extract text and measures /uni sequence corruption.
"""
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader


def detect_text_quality(text: str) -> dict:
    """Detect text quality issues like font encoding problems."""
    if not text:
        return {"quality": "empty", "corrupted_ratio": 1.0, "uni_sequences": 0}
    
    # Count /uni escape sequences (indicates font encoding issues)
    uni_pattern = r'/uni[0-9a-fA-F]{8}'
    uni_matches = re.findall(uni_pattern, text)
    uni_count = len(uni_matches)
    
    total_chars = len(text)
    uni_corrupted_chars = uni_count * 12  # Each /uni sequence is 12 chars
    
    corrupted_ratio = min(1.0, uni_corrupted_chars / max(1, total_chars))
    
    return {
        "uni_sequences": uni_count,
        "corrupted_ratio": round(corrupted_ratio, 4),
        "total_chars": total_chars,
    }


def extract_pdf_text(pdf_path: str, max_pages: int = 10) -> str:
    """Extract text from first N pages of PDF."""
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for i, page in enumerate(reader.pages[:max_pages]):
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception as e:
        return f"ERROR: {e}"


def extract_arxiv_id(filename: str) -> str:
    """Extract arxiv ID from filename (e.g., 2405.10938v3.Title.pdf -> 2405.10938)"""
    # Match pattern like 2405.10938v3 or 2405.10938
    match = re.match(r'(\d{4}\.\d{4,5})(v\d+)?', filename)
    if match:
        return match.group(1)  # Return just the arxiv ID without version
    return filename.split(".")[0]


def analyze_pdf(pdf_path: str) -> dict:
    """Analyze a single PDF for quality."""
    text = extract_pdf_text(pdf_path)
    if text.startswith("ERROR:"):
        return {"error": text, "arxiv_id": extract_arxiv_id(Path(pdf_path).stem)}
    
    quality = detect_text_quality(text)
    
    # Extract arxiv ID from filename
    filename = Path(pdf_path).stem
    arxiv_id = extract_arxiv_id(filename)
    
    return {
        "arxiv_id": arxiv_id,
        "pdf_path": pdf_path,
        "uni_sequences": quality["uni_sequences"],
        "corrupted_ratio": quality["corrupted_ratio"],
        "total_chars": quality["total_chars"],
    }


def main():
    downloads_dir = Path(__file__).parent.parent / "storage" / "downloads"
    pdf_files = list(downloads_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDFs", flush=True)
    print("Analyzing PDFs for quality...", flush=True)
    
    results = []
    for i, pdf_path in enumerate(pdf_files):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(pdf_files)}", flush=True)
        result = analyze_pdf(str(pdf_path))
        results.append(result)
    
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    print(f"\nValid results: {len(valid_results)}/{len(results)}", flush=True)
    
    # Sort by corrupted_ratio (highest first = worst quality)
    sorted_by_corruption = sorted(valid_results, key=lambda x: x["corrupted_ratio"], reverse=True)
    
    # Get 50 worst (bad quality)
    bad_papers = sorted_by_corruption[:50]
    
    # Get 50 best (good quality) from the end
    good_papers = sorted_by_corruption[-50:]
    
    # Create config
    config = {
        "bad_papers": [
            {
                "arxiv_id": p["arxiv_id"],
                "pdf_path": p["pdf_path"],
                "corrupted_ratio": p["corrupted_ratio"],
                "uni_sequences": p["uni_sequences"],
            }
            for p in bad_papers
        ],
        "good_papers": [
            {
                "arxiv_id": p["arxiv_id"],
                "pdf_path": p["pdf_path"],
                "corrupted_ratio": p["corrupted_ratio"],
                "uni_sequences": p["uni_sequences"],
            }
            for p in good_papers
        ],
        "extractors": ["pypdf", "pymupdf", "pdfplumber", "pdfminer", "deepseek-ocr-2"],
        "chunk_size": 5000,
        "output_file": "storage/schemas/pdf_extraction_results.json",
    }
    
    # Save config
    output_path = Path(__file__).parent / "test_pdf_config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig saved to: {output_path}", flush=True)
    print(f"\nBad papers (top 5 by corruption):", flush=True)
    for p in bad_papers[:5]:
        print(f"  {p['arxiv_id']}: {p['corrupted_ratio']:.2%} corrupted, {p['uni_sequences']} /uni sequences", flush=True)
    
    print(f"\nGood papers (top 5 by quality):", flush=True)
    for p in good_papers[-5:]:
        print(f"  {p['arxiv_id']}: {p['corrupted_ratio']:.2%} corrupted, {p['uni_sequences']} /uni sequences", flush=True)


if __name__ == "__main__":
    main()
