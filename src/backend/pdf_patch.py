"""
Monkey-patch pypdf to use pymupdf (fitz) for text extraction.

This patch replaces pypdf.PdfReader with a pymupdf-backed implementation
that produces cleaner text extraction, especially for PDFs with Type3 fonts.

Usage:
    from pdf_patch import patch_pypdf
    patch_pypdf()  # Call before importing paperqa or any module that uses pypdf
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, BinaryIO, Union

import fitz  # pymupdf

logger = logging.getLogger(__name__)


class PyMuPDFPage:
    """Wrapper to make pymupdf page act like pypdf page."""
    
    def __init__(self, page: fitz.Page):
        self._page = page
    
    def extract_text(self, *args: Any, **kwargs: Any) -> str:
        """Extract text from page using pymupdf."""
        return self._page.get_text()


class PyMuPDFReader:
    """Drop-in replacement for pypdf.PdfReader using pymupdf.
    
    This class mimics the pypdf.PdfReader interface but uses pymupdf
    for actual PDF parsing and text extraction.
    """
    
    def __init__(
        self,
        stream: Union[str, Path, BinaryIO],
        strict: bool = False,
        password: str | None = None,
    ):
        """Initialize the PDF reader.
        
        Args:
            stream: File path or file-like object
            strict: Ignored (for compatibility)
            password: PDF password if encrypted
        """
        self._doc: fitz.Document | None = None
        self._pages: list[PyMuPDFPage] = []
        
        try:
            if isinstance(stream, (str, Path)):
                self._doc = fitz.open(str(stream))
            else:
                # File-like object - read bytes
                data = stream.read()
                self._doc = fitz.open(stream=data, filetype="pdf")
            
            if password and self._doc.is_encrypted:
                self._doc.authenticate(password)
            
            self._pages = [PyMuPDFPage(page) for page in self._doc]
            
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
    
    @property
    def pages(self) -> list[PyMuPDFPage]:
        """Return list of pages."""
        return self._pages
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Return PDF metadata."""
        if self._doc is None:
            return {}
        return dict(self._doc.metadata) if self._doc.metadata else {}
    
    def __len__(self) -> int:
        """Return number of pages."""
        return len(self._pages)
    
    def __del__(self) -> None:
        """Clean up resources."""
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:
                pass
    
    def close(self) -> None:
        """Close the PDF document."""
        if self._doc is not None:
            self._doc.close()
            self._doc = None


# Store original PdfReader for potential restoration
_original_pdf_reader = None
_patched = False


def patch_pypdf() -> None:
    """Replace pypdf.PdfReader with pymupdf-backed reader.
    
    This function should be called early in the application startup,
    before any module imports pypdf.PdfReader.
    """
    global _original_pdf_reader, _patched
    
    if _patched:
        logger.debug("pypdf already patched with pymupdf")
        return
    
    try:
        import pypdf
        _original_pdf_reader = pypdf.PdfReader
        pypdf.PdfReader = PyMuPDFReader
        _patched = True
        logger.info("Successfully patched pypdf.PdfReader with pymupdf backend")
    except ImportError:
        logger.warning("pypdf not installed, skipping patch")


def unpatch_pypdf() -> None:
    """Restore original pypdf.PdfReader."""
    global _original_pdf_reader, _patched
    
    if not _patched or _original_pdf_reader is None:
        return
    
    try:
        import pypdf
        pypdf.PdfReader = _original_pdf_reader
        _patched = False
        logger.info("Restored original pypdf.PdfReader")
    except ImportError:
        pass


def is_patched() -> bool:
    """Check if pypdf is currently patched."""
    return _patched
