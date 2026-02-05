# PDF Extraction Quality Test

This directory contains scripts and results for comparing PDF text extraction quality across multiple libraries.

## Background

The paper-curator pipeline was experiencing text extraction issues where some PDFs produced corrupted text with `/uniXXXXXXXX` escape sequences instead of readable Unicode characters. This test suite was created to identify the best extraction library.

## Files

| File | Description |
|------|-------------|
| `identify_test_pdfs.py` | Analyzes all PDFs and identifies 50 worst + 50 best quality samples |
| `test_pdf_extraction.py` | Main test script comparing 4 extractors on 100 PDFs |
| `test_pdf_config.json` | Test configuration with PDF paths and expected quality |
| `pdf_extraction_results.json` | Full extraction test results |
| `test_deepseek_ocr.pbs` | PBS job script for DeepSeek-OCR-2 (optional GPU test) |
| `test_deepseek_ocr_extraction.py` | DeepSeek-OCR-2 extraction script (optional) |

## Quick Start

```bash
# Activate uv environment
uvenv infer

# Step 1: Identify test PDFs (creates test_pdf_config.json)
python scripts/pdf_extraction_test/identify_test_pdfs.py

# Step 2: Run extraction test (10 chunks per PDF, 100 PDFs, 4 extractors)
python scripts/pdf_extraction_test/test_pdf_extraction.py

# Optional: Run DeepSeek-OCR-2 test on GPU
qsub scripts/pdf_extraction_test/test_deepseek_ocr.pbs
```

## Test Methodology

1. **Sample Selection**: 100 PDFs (50 worst quality by pypdf, 50 best quality)
2. **Chunk Sampling**: 10 chunks per PDF, evenly distributed (0%, 11%, 22%, ..., 100%)
3. **Chunk Size**: 5000 characters per chunk
4. **Total Analysis**: 4000 chunks (100 PDFs × 10 chunks × 4 extractors)

### Quality Metric

Each chunk is classified as:
- **Good**: < 30% corrupted characters (no `/uni` sequences)
- **Bad**: >= 30% corrupted characters

Quality score = good_chunks / total_chunks

## Test Results Summary

**Run Date**: 2026-02-05  
**Configuration**: 100 PDFs × 10 chunks × 4 extractors = 4000 total chunk evaluations

| Extractor | Avg Quality | Good/Bad Chunks | Uni Sequences | Avg Time |
|-----------|-------------|-----------------|---------------|----------|
| pypdf | 94.60% | 946/54 | 17,060 | 0.953s |
| **pymupdf** | **100.00%** | **1000/0** | **0** | **0.167s** |
| pdfplumber | 100.00% | 1000/0 | 0 | 4.131s |
| pdfminer | 100.00% | 1000/0 | 0 | 2.834s |

### Bad Paper Recovery

Out of 50 papers with known extraction issues (>30% corruption with pypdf):

| Extractor | Papers Recovered to >70% Quality |
|-----------|----------------------------------|
| pypdf | 42/50 (84%) |
| **pymupdf** | **50/50 (100%)** |
| pdfplumber | 50/50 (100%) |
| pdfminer | 50/50 (100%) |

### Performance Comparison

| Extractor | Relative Speed | Notes |
|-----------|----------------|-------|
| **pymupdf** | **1.0x (fastest)** | Best choice |
| pypdf | 5.7x slower | Has quality issues |
| pdfminer | 17.0x slower | Perfect quality |
| pdfplumber | 24.7x slower | Perfect quality |

## Sample Per-Paper Results (Worst Case: 2405.10938)

This paper had 40.71% corruption with pypdf:

| Extractor | Total Chars | Good/Bad Chunks | Quality | Time |
|-----------|-------------|-----------------|---------|------|
| pypdf | 491,524 | 4/10 | 40% | 2.84s |
| pymupdf | 158,256 | 10/10 | 100% | 0.75s |
| pdfplumber | 182,630 | 10/10 | 100% | 11.8s |
| pdfminer | 187,688 | 10/10 | 100% | 8.32s |

Key observations:
- pypdf extracts 3x more characters (most are garbage `/uni` sequences)
- pymupdf is fastest AND produces cleanest output
- All alternatives achieve 100% quality on the worst case

## Conclusion

**Winner: pymupdf (fitz)**

- 100% extraction quality on all 1000 sampled chunks
- 5.7x faster than pypdf, 17-25x faster than alternatives
- Zero `/uni` escape sequences detected
- No GPU required

## Recommended Mitigation

Replace pypdf with pymupdf in the PaperQA pipeline:

```python
# Before (pypdf - problematic)
from pypdf import PdfReader
reader = PdfReader(pdf_path)
text = "\n".join(page.extract_text() for page in reader.pages)

# After (pymupdf - recommended)
import fitz
doc = fitz.open(pdf_path)
text = "\n".join(page.get_text() for page in doc)
doc.close()
```
