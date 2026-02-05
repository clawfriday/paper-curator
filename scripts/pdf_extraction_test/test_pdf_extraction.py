#!/usr/bin/env python3
"""
Test PDF extraction quality across multiple libraries.
Compares pypdf, pymupdf, pdfplumber, and pdfminer on 100 test PDFs.
Samples exactly 10 chunks per PDF at consistent positions for fair comparison.
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# PDF extraction libraries
from pypdf import PdfReader
import fitz  # pymupdf
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract


def detect_chunk_quality(text: str) -> dict:
    """
    Detect chunk quality and return detailed metrics.
    Returns dict with is_good (bool) and corruption details.
    """
    if not text or len(text.strip()) < 100:
        return {
            "is_good": False,
            "uni_sequences": 0,
            "corrupted_ratio": 1.0,
            "reason": "empty_or_too_short",
        }
    
    # Count /uni escape sequences (indicates font encoding issues)
    uni_pattern = r'/uni[0-9a-fA-F]{8}'
    uni_matches = re.findall(uni_pattern, text)
    uni_count = len(uni_matches)
    
    total_chars = len(text)
    uni_corrupted_chars = uni_count * 12  # Each /uni sequence is 12 chars
    
    corrupted_ratio = uni_corrupted_chars / max(1, total_chars)
    
    # Good if less than 30% corrupted
    is_good = corrupted_ratio < 0.3
    
    return {
        "is_good": is_good,
        "uni_sequences": uni_count,
        "corrupted_ratio": round(corrupted_ratio, 4),
        "reason": "clean" if is_good else "corrupted",
    }


def sample_chunks(text: str, n_chunks: int = 10, chunk_size: int = 5000) -> list[dict]:
    """
    Sample n_chunks evenly distributed across the document.
    Each chunk is chunk_size characters.
    Returns list of dicts with chunk text and position info.
    """
    if not text or len(text) < chunk_size:
        return []
    
    total_len = len(text)
    # Calculate positions for evenly distributed sampling
    # If document is 100k chars and we want 10 chunks of 5k each,
    # sample at positions: 0%, 10%, 20%, ... 90% of (total - chunk_size)
    
    available_range = total_len - chunk_size
    if available_range <= 0:
        # Document shorter than chunk_size, return single chunk
        return [{
            "position": 0,
            "position_pct": 0.0,
            "text": text[:chunk_size],
        }]
    
    chunks = []
    for i in range(n_chunks):
        # Calculate position as percentage through document
        pct = i / max(1, n_chunks - 1) if n_chunks > 1 else 0
        position = int(available_range * pct)
        
        chunk_text = text[position:position + chunk_size]
        chunks.append({
            "position": position,
            "position_pct": round(pct, 2),
            "text": chunk_text,
        })
    
    return chunks


class PDFExtractor:
    """Base class for PDF extraction methods."""
    name: str = "base"
    
    def extract(self, pdf_path: str) -> str:
        """Extract full text from PDF. Returns error string on failure."""
        raise NotImplementedError


class PyPDFExtractor(PDFExtractor):
    name = "pypdf"
    
    def extract(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception as e:
            return f"ERROR: {e}"


class PyMuPDFExtractor(PDFExtractor):
    name = "pymupdf"
    
    def extract(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            return f"ERROR: {e}"


class PDFPlumberExtractor(PDFExtractor):
    name = "pdfplumber"
    
    def extract(self, pdf_path: str) -> str:
        try:
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    text_parts.append(text)
            return "\n".join(text_parts)
        except Exception as e:
            return f"ERROR: {e}"


class PDFMinerExtractor(PDFExtractor):
    name = "pdfminer"
    
    def extract(self, pdf_path: str) -> str:
        try:
            return pdfminer_extract(pdf_path)
        except Exception as e:
            return f"ERROR: {e}"


def run_extraction_test(
    pdf_path: str,
    arxiv_id: str,
    expected_quality: str,
    extractors: list[PDFExtractor],
    n_chunks: int = 10,
    chunk_size: int = 5000,
) -> dict[str, Any]:
    """Run extraction test on a single PDF with all extractors."""
    result = {
        "arxiv_id": arxiv_id,
        "pdf_path": pdf_path,
        "expected_quality": expected_quality,
        "n_chunks_sampled": n_chunks,
        "chunk_size": chunk_size,
        "extractors": {},
    }
    
    for extractor in extractors:
        start_time = time.time()
        text = extractor.extract(pdf_path)
        extraction_time = time.time() - start_time
        
        if text.startswith("ERROR:"):
            result["extractors"][extractor.name] = {
                "error": text,
                "extraction_time_s": round(extraction_time, 3),
            }
            continue
        
        # Sample exactly n_chunks evenly distributed
        chunks = sample_chunks(text, n_chunks, chunk_size)
        
        # Assess quality of each chunk
        chunk_results = []
        good_count = 0
        bad_count = 0
        total_uni_sequences = 0
        
        for chunk in chunks:
            quality = detect_chunk_quality(chunk["text"])
            chunk_results.append({
                "position_pct": chunk["position_pct"],
                "is_good": quality["is_good"],
                "uni_sequences": quality["uni_sequences"],
                "corrupted_ratio": quality["corrupted_ratio"],
            })
            if quality["is_good"]:
                good_count += 1
            else:
                bad_count += 1
            total_uni_sequences += quality["uni_sequences"]
        
        quality_score = good_count / max(1, len(chunks)) if chunks else 0
        
        result["extractors"][extractor.name] = {
            "total_chars": len(text),
            "chunks_sampled": len(chunks),
            "good_chunks": good_count,
            "bad_chunks": bad_count,
            "quality_score": round(quality_score, 4),
            "total_uni_sequences": total_uni_sequences,
            "extraction_time_s": round(extraction_time, 3),
            "chunk_details": chunk_results,
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Test PDF extraction quality")
    parser.add_argument(
        "--config",
        default="scripts/pdf_extraction_test/test_pdf_config.json",
        help="Path to test configuration JSON",
    )
    parser.add_argument(
        "--extractors",
        default="pypdf,pymupdf,pdfplumber,pdfminer",
        help="Comma-separated list of extractors to test",
    )
    parser.add_argument(
        "--output",
        default="scripts/pdf_extraction_test/pdf_extraction_results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=10,
        help="Number of chunks to sample per PDF (default: 10)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Size of each chunk in characters (default: 5000)",
    )
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent.parent / args.config
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Select extractors
    extractor_map = {
        "pypdf": PyPDFExtractor(),
        "pymupdf": PyMuPDFExtractor(),
        "pdfplumber": PDFPlumberExtractor(),
        "pdfminer": PDFMinerExtractor(),
    }
    
    selected_extractors = [
        extractor_map[name]
        for name in args.extractors.split(",")
        if name in extractor_map
    ]
    
    print(f"Testing extractors: {[e.name for e in selected_extractors]}", flush=True)
    print(f"Chunks per PDF: {args.n_chunks}", flush=True)
    print(f"Chunk size: {args.chunk_size} chars", flush=True)
    
    # Collect all papers
    all_papers = []
    for paper in config["bad_papers"]:
        paper["expected_quality"] = "bad"
        all_papers.append(paper)
    for paper in config["good_papers"]:
        paper["expected_quality"] = "good"
        all_papers.append(paper)
    
    print(f"Total papers to test: {len(all_papers)}", flush=True)
    print(f"Total chunks to analyze: {len(all_papers) * args.n_chunks * len(selected_extractors)}", flush=True)
    
    # Run tests
    results = {}
    for i, paper in enumerate(all_papers):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(all_papers)} PDFs processed", flush=True)
        
        result = run_extraction_test(
            paper["pdf_path"],
            paper["arxiv_id"],
            paper["expected_quality"],
            selected_extractors,
            args.n_chunks,
            args.chunk_size,
        )
        results[paper["arxiv_id"]] = result
    
    print(f"Progress: {len(all_papers)}/{len(all_papers)} PDFs processed", flush=True)
    
    # Calculate summary statistics
    summary = {}
    for extractor in selected_extractors:
        scores = []
        total_good = 0
        total_bad = 0
        total_uni = 0
        total_time = 0
        improved_bad = 0
        errors = 0
        
        for arxiv_id, result in results.items():
            ext_result = result["extractors"].get(extractor.name, {})
            if "error" in ext_result:
                errors += 1
                continue
            
            score = ext_result.get("quality_score", 0)
            scores.append(score)
            total_good += ext_result.get("good_chunks", 0)
            total_bad += ext_result.get("bad_chunks", 0)
            total_uni += ext_result.get("total_uni_sequences", 0)
            total_time += ext_result.get("extraction_time_s", 0)
            
            # Count improved bad papers (originally bad, now >70% good)
            if result["expected_quality"] == "bad" and score > 0.7:
                improved_bad += 1
        
        total_chunks = total_good + total_bad
        summary[extractor.name] = {
            "avg_quality_score": round(sum(scores) / max(1, len(scores)), 4),
            "min_quality_score": round(min(scores) if scores else 0, 4),
            "max_quality_score": round(max(scores) if scores else 0, 4),
            "total_chunks_analyzed": total_chunks,
            "total_good_chunks": total_good,
            "total_bad_chunks": total_bad,
            "total_uni_sequences": total_uni,
            "improved_bad_papers": improved_bad,
            "total_bad_papers": len(config["bad_papers"]),
            "total_extraction_time_s": round(total_time, 2),
            "avg_extraction_time_s": round(total_time / max(1, len(scores)), 3),
            "errors": errors,
        }
    
    # Build output
    output = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "total_pdfs": len(all_papers),
            "chunks_per_pdf": args.n_chunks,
            "chunk_size": args.chunk_size,
            "extractors_tested": [e.name for e in selected_extractors],
        },
        "results": results,
        "summary": summary,
    }
    
    # Save output
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent.parent / args.output
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}", flush=True)
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    
    # Print comparison table
    print(f"\n{'Extractor':<12} | {'Avg Quality':>11} | {'Good/Bad Chunks':>16} | {'Uni Seqs':>10} | {'Avg Time':>10}", flush=True)
    print("-"*70, flush=True)
    for ext_name, stats in summary.items():
        chunks_str = f"{stats['total_good_chunks']}/{stats['total_bad_chunks']}"
        print(f"{ext_name:<12} | {stats['avg_quality_score']:>10.2%} | {chunks_str:>16} | {stats['total_uni_sequences']:>10} | {stats['avg_extraction_time_s']:>9.3f}s", flush=True)
    
    print("\n" + "-"*70, flush=True)
    print("Bad Paper Recovery (50 papers with known extraction issues):", flush=True)
    for ext_name, stats in summary.items():
        print(f"  {ext_name}: {stats['improved_bad_papers']}/{stats['total_bad_papers']} papers recovered to >70% quality", flush=True)


if __name__ == "__main__":
    main()
