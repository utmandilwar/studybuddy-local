"""Tests for DocumentLoader validating PDF/TXT ingestion and error handling."""

import fitz  # PyMuPDF
import pytest

from src.document_loader import DocumentLoader
from src.exceptions import (
    DocumentExtractionError,
    EmptyDocumentError,
    OversizedFileError,
    UnsupportedFileTypeError,
)


@pytest.fixture
def valid_pdf_bytes() -> bytes:
    """Create a minimal valid PDF in memory with known text."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Chapter 1: Operating Systems Basics and Architecture.")
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Chapter 2: Memory Management, Paging, and Segmentation.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """Create a valid PDF with blank pages (no extractable text)."""
    doc = fitz.open()
    doc.new_page()  # blank page
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_load_valid_txt():
    """Valid text files extract with 1 page content."""
    loader = DocumentLoader()
    content = "Summary of Operating Systems\nProcesses and threads."
    pages = loader.load_from_bytes(content.encode("utf-8"), "notes.txt")
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Summary of Operating Systems" in pages[0].text


def test_load_empty_txt():
    """Empty text files raise EmptyDocumentError."""
    loader = DocumentLoader()
    with pytest.raises(EmptyDocumentError):
        loader.load_from_bytes(b"", "empty.txt")

    with pytest.raises(EmptyDocumentError):
        loader.load_from_bytes(b"   \n  \t  ", "whitespace.txt")


def test_load_valid_pdf(valid_pdf_bytes):
    """Valid multi-page PDF extracts with correct page numbers and text."""
    loader = DocumentLoader()
    pages = loader.load_from_bytes(valid_pdf_bytes, "os_notes.pdf")
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Chapter 1" in pages[0].text
    assert pages[1].page_number == 2
    assert "Chapter 2" in pages[1].text


def test_load_empty_scanned_pdf(empty_pdf_bytes):
    """PDF with no extractable text raises EmptyDocumentError advising on OCR."""
    loader = DocumentLoader()
    with pytest.raises(EmptyDocumentError) as exc_info:
        loader.load_from_bytes(empty_pdf_bytes, "scanned_doc.pdf")
    assert "scanned document" in str(exc_info.value).lower() or "no extractable text" in str(exc_info.value).lower()


def test_load_corrupted_pdf():
    """Corrupted PDF bytes raise DocumentExtractionError."""
    loader = DocumentLoader()
    corrupted_bytes = b"%PDF-1.4 completely invalid corrupt binary garbage"
    with pytest.raises(DocumentExtractionError):
        loader.load_from_bytes(corrupted_bytes, "corrupt.pdf")


def test_load_unsupported_extension():
    """Files with unsupported extensions raise UnsupportedFileTypeError."""
    loader = DocumentLoader()
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        loader.load_from_bytes(b"some content", "homework.docx")
    assert "unsupported file type" in str(exc_info.value).lower()


def test_load_oversized_file():
    """Files exceeding size limit raise OversizedFileError."""
    loader = DocumentLoader(max_size_mb=1)  # 1 MB limit
    oversized_bytes = b"X" * (1024 * 1024 + 100)
    with pytest.raises(OversizedFileError) as exc_info:
        loader.load_from_bytes(oversized_bytes, "large_notes.txt")
    assert "exceeds maximum upload limit" in str(exc_info.value)
