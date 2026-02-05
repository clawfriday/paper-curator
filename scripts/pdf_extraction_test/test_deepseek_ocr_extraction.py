#!/usr/bin/env python3
"""
Test DeepSeek-OCR-2 extraction quality on problematic PDFs.
Uses transformers to run the model on GPU.
"""
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from pdf2image import convert_from_path
from PIL import Image


def detect_chunk_quality(text: str) -> bool:
    """
    Detect if a chunk is good quality.
    Returns True if good, False if bad (>30% corrupted).
    """
    if not text:
        return False
    
    # Count /uni escape sequences (indicates font encoding issues)
    uni_pattern = r'/uni[0-9a-fA-F]{8}'
    uni_matches = re.findall(uni_pattern, text)
    uni_count = len(uni_matches)
    
    total_chars = len(text)
    uni_corrupted_chars = uni_count * 12
    
    corrupted_ratio = uni_corrupted_chars / max(1, total_chars)
    return corrupted_ratio < 0.3


def chunk_text(text: str, chunk_size: int = 5000) -> list[str]:
    """Split text into chunks of specified size."""
    if not text:
        return []
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


class DeepSeekOCR2Extractor:
    """DeepSeek-OCR-2 extractor using transformers."""
    
    name = "deepseek-ocr-2"
    
    def __init__(self):
        self.model = None
        self.processor = None
    
    def load_model(self):
        """Load the DeepSeek-OCR-2 model."""
        if self.model is not None:
            return
        
        from transformers import AutoModel
        
        print("Loading DeepSeek-OCR-2 model...", flush=True)
        self.model = AutoModel.from_pretrained(
            "deepseek-ai/DeepSeek-OCR-2",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded.", flush=True)
    
    def extract(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR."""
        try:
            self.load_model()
            
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=150)
            
            text_parts = []
            for i, image in enumerate(images):
                # Run OCR on each page
                result = self.model.ocr(image)
                text_parts.append(result or "")
            
            return "\n\n".join(text_parts)
        except Exception as e:
            return f"ERROR: {e}"


def main():
    # Load config
    config_path = Path(__file__).parent / "test_pdf_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Only test on bad papers (where OCR might help)
    bad_papers = config["bad_papers"][:10]  # Test on first 10 bad papers for speed
    
    print(f"Testing DeepSeek-OCR-2 on {len(bad_papers)} papers", flush=True)
    
    extractor = DeepSeekOCR2Extractor()
    chunk_size = config["chunk_size"]
    
    results = {}
    for i, paper in enumerate(bad_papers):
        print(f"\nProgress: {i+1}/{len(bad_papers)}", flush=True)
        print(f"Processing: {paper['arxiv_id']}", flush=True)
        
        start_time = time.time()
        text = extractor.extract(paper["pdf_path"])
        extraction_time = time.time() - start_time
        
        if text.startswith("ERROR:"):
            results[paper["arxiv_id"]] = {
                "error": text,
                "extraction_time_s": round(extraction_time, 3),
            }
            continue
        
        # Chunk and assess quality
        chunks = chunk_text(text, chunk_size)
        good_chunks = sum(1 for c in chunks if detect_chunk_quality(c))
        bad_chunks = len(chunks) - good_chunks
        
        quality_score = good_chunks / max(1, len(chunks))
        
        results[paper["arxiv_id"]] = {
            "total_chunks": len(chunks),
            "good_chunks": good_chunks,
            "bad_chunks": bad_chunks,
            "quality_score": round(quality_score, 4),
            "extraction_time_s": round(extraction_time, 3),
            "total_chars": len(text),
            "text_sample": text[:500] if text else "",
        }
        
        print(f"  Quality: {quality_score:.2%}, Time: {extraction_time:.1f}s", flush=True)
    
    # Save results
    output = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "model": "deepseek-ai/DeepSeek-OCR-2",
            "total_papers_tested": len(bad_papers),
        },
        "results": results,
    }
    
    output_path = Path(__file__).parent.parent / "storage/schemas/deepseek_ocr_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}", flush=True)
    
    # Print summary
    scores = [r["quality_score"] for r in results.values() if "quality_score" in r]
    if scores:
        print(f"\nSummary:", flush=True)
        print(f"  Average quality: {sum(scores)/len(scores):.2%}", flush=True)
        print(f"  Min quality: {min(scores):.2%}", flush=True)
        print(f"  Max quality: {max(scores):.2%}", flush=True)


if __name__ == "__main__":
    main()
