"""Tests for TextChunker validating boundaries, overlaps, and metadata preservation."""

import pytest

from src.chunker import TextChunker
from src.document_loader import PageContent


def test_chunker_empty_input():
    """Empty or whitespace text returns no chunks."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\t  ") == []


def test_chunker_short_text():
    """Text shorter than chunk size produces exactly one chunk."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    text = "Short study note about CPU registers and program counter."
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunker_long_text_and_overlap():
    """Text longer than chunk size splits into multiple overlapping pieces."""
    chunker = TextChunker(chunk_size=60, chunk_overlap=15)
    text = (
        "Operating systems coordinate CPU operations. "
        "They manage system memory and schedule running tasks. "
        "Device drivers handle hardware communication. "
        "File systems organize disk blocks and metadata."
    )
    chunks = chunker.split_text(text)
    assert len(chunks) > 1

    # Verify that each chunk respects length
    for chunk in chunks:
        assert len(chunk) <= 75  # within boundary tolerance


def test_chunker_invalid_overlap_raises():
    """Overlap equal to or greater than chunk size raises ValueError."""
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunk_pages_preserves_metadata():
    """Chunking PageContent preserves page numbers, doc_id, and sequential chunk indices."""
    chunker = TextChunker(chunk_size=80, chunk_overlap=20)
    pages = [
        PageContent(page_number=1, text="Page 1 discussion of thread concurrency and semaphores."),
        PageContent(page_number=2, text="Page 2 analysis of deadlock conditions and mutual exclusion."),
    ]

    doc_chunks = chunker.chunk_pages(pages, doc_id="doc_os101", document_name="os101.pdf")
    assert len(doc_chunks) >= 2

    # Check first chunk
    assert doc_chunks[0].doc_id == "doc_os101"
    assert doc_chunks[0].document_name == "os101.pdf"
    assert doc_chunks[0].page_number == 1
    assert doc_chunks[0].chunk_index == 0
    assert doc_chunks[0].chunk_id == "doc_os101_p1_c0"

    # Check subsequent chunk has sequential index
    assert doc_chunks[1].chunk_index == 1
    assert doc_chunks[1].chunk_id == f"doc_os101_p{doc_chunks[1].page_number}_c1"
