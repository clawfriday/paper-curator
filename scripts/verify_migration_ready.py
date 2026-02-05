#!/usr/bin/env python3
"""Verify everything is ready for pymupdf migration."""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

print("=" * 60)
print("pymupdf Migration Readiness Check")
print("=" * 60)

checks_passed = 0
checks_failed = 0

# Check 1: pdf_patch module exists
print("\n1. Checking pdf_patch module...")
try:
    from pdf_patch import patch_pypdf, is_patched, PyMuPDFReader
    print("   ✓ pdf_patch module imported successfully")
    checks_passed += 1
except ImportError as e:
    print(f"   ✗ Failed to import pdf_patch: {e}")
    checks_failed += 1

# Check 2: pymupdf installed
print("\n2. Checking pymupdf (fitz) installation...")
try:
    import fitz
    print(f"   ✓ pymupdf version: {fitz.version}")
    checks_passed += 1
except ImportError:
    print("   ✗ pymupdf not installed")
    checks_failed += 1

# Check 3: Patch works
print("\n3. Testing patch mechanism...")
try:
    patch_pypdf()
    if is_patched():
        print("   ✓ Patch applied successfully")
        checks_passed += 1
    else:
        print("   ✗ Patch did not apply")
        checks_failed += 1
except Exception as e:
    print(f"   ✗ Patch failed: {e}")
    checks_failed += 1

# Check 4: pypdf uses patched reader
print("\n4. Verifying pypdf.PdfReader is patched...")
try:
    import pypdf
    if pypdf.PdfReader.__name__ == "PyMuPDFReader":
        print("   ✓ pypdf.PdfReader = PyMuPDFReader")
        checks_passed += 1
    else:
        print(f"   ✗ pypdf.PdfReader = {pypdf.PdfReader.__name__}")
        checks_failed += 1
except Exception as e:
    print(f"   ✗ Error: {e}")
    checks_failed += 1

# Check 5: Test PDF extraction
print("\n5. Testing PDF extraction on sample...")
test_config = Path(__file__).parent / "pdf_extraction_test" / "test_pdf_config.json"
if test_config.exists():
    import json
    import re
    with open(test_config) as f:
        config = json.load(f)
    
    bad_pdf = config['bad_papers'][0]
    pdf_path = bad_pdf['pdf_path']
    
    if Path(pdf_path).exists():
        reader = pypdf.PdfReader(pdf_path)
        text = ''.join(page.extract_text() for page in reader.pages)
        uni_count = len(re.findall(r'/uni[0-9a-fA-F]{8}', text))
        
        if uni_count == 0:
            print(f"   ✓ PDF {bad_pdf['arxiv_id']} extracted cleanly (0 /uni sequences)")
            print(f"     Original corruption: {bad_pdf['corrupted_ratio']:.1%}")
            checks_passed += 1
        else:
            print(f"   ✗ PDF still has {uni_count} /uni sequences")
            checks_failed += 1
    else:
        print(f"   - PDF not found: {pdf_path}")
else:
    print(f"   - Test config not found: {test_config}")

# Check 6: Database connection
print("\n6. Checking database connection...")
try:
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        database=os.environ.get("PGDATABASE", "paper_curator"),
        user=os.environ.get("PGUSER", "curator"),
        password=os.environ.get("PGPASSWORD", "curator123"),
    )
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM papers")
        paper_count = cur.fetchone()[0]
    conn.close()
    print(f"   ✓ Database connected ({paper_count} papers in DB)")
    checks_passed += 1
except Exception as e:
    print(f"   ✗ Database connection failed: {e}")
    print("     (This is OK if running locally without DB)")
    # Don't count as failure for local testing

# Check 7: PaperQA indices
print("\n7. Checking PaperQA index directory...")
index_dir = Path(__file__).parent.parent / "storage" / "paperqa_index"
if index_dir.exists():
    pkl_files = list(index_dir.glob("*.pkl"))
    print(f"   Found {len(pkl_files)} index files")
    if pkl_files:
        print(f"   Total size: {sum(f.stat().st_size for f in pkl_files) / 1024 / 1024:.1f} MB")
    checks_passed += 1
else:
    print(f"   - Index directory not found: {index_dir}")

# Summary
print("\n" + "=" * 60)
print(f"Results: {checks_passed} passed, {checks_failed} failed")
print("=" * 60)

if checks_failed == 0:
    print("\n✓ All checks passed! Ready for migration.")
    print("\nTo run migration:")
    print("  1. Start backend: make singularity-run")
    print("  2. Run migration: python scripts/migrate_to_pymupdf.py --dry-run")
    print("  3. If dry-run looks good: python scripts/migrate_to_pymupdf.py")
else:
    print("\n✗ Some checks failed. Please fix issues before migration.")
    sys.exit(1)
