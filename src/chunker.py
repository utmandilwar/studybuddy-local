"""Document chunking with sentence/paragraph boundary awareness for StudyBuddy Local.

Preserves document identifier, filename, page numbers, and chunk indices.
"""

from __future__ import annotations

import re
from typing import List, Optional

from config import AppConfig
from src.document_loader import PageContent
from src.schemas import DocumentChunk


class TextChunker:
    """Splits document text into overlapping chunks respecting sentence boundaries."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or AppConfig.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or AppConfig.CHUNK_OVERLAP
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

    def split_text(self, text: str) -> List[str]:
        """Split a string into overlapping chunks, attempting sentence boundaries.

        Args:
            text: Plain text to split.

        Returns:
            List of chunk strings.
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Look for paragraph or sentence break near the end of the window
            lookback_window = min(self.chunk_overlap * 2, self.chunk_size // 2)
            search_start = max(start, end - lookback_window)
            substring = text[search_start:end]

            # Try finding paragraph break (\n\n) first, then newline, then sentence end (. / ? / !)
            split_pos = -1
            match_para = [m.start() for m in re.finditer(r"\n\s*\n", substring)]
            if match_para:
                split_pos = search_start + match_para[-1] + 2
            else:
                match_sent = [m.start() for m in re.finditer(r"[.!?]\s+", substring)]
                if match_sent:
                    split_pos = search_start + match_sent[-1] + 2
                else:
                    match_line = [m.start() for m in re.finditer(r"\n", substring)]
                    if match_line:
                        split_pos = search_start + match_line[-1] + 1
                    else:
                        match_space = [m.start() for m in re.finditer(r"\s+", substring)]
                        if match_space:
                            split_pos = search_start + match_space[-1] + 1

            if split_pos > start:
                chunk_str = text[start:split_pos].strip()
                if chunk_str:
                    chunks.append(chunk_str)
                # Next chunk starts with overlap
                start = max(split_pos - self.chunk_overlap, start + 1)
            else:
                # Fallback to hard character boundary
                chunk_str = text[start:end].strip()
                if chunk_str:
                    chunks.append(chunk_str)
                start = end - self.chunk_overlap

        return [c for c in chunks if c]

    def chunk_pages(
        self,
        pages: List[PageContent],
        doc_id: str,
        document_name: str,
    ) -> List[DocumentChunk]:
        """Convert a list of PageContent objects into structured DocumentChunks.

        Args:
            pages: List of extracted document pages.
            doc_id: Unique document identifier.
            document_name: Original file name.

        Returns:
            List of DocumentChunk schema instances.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            split_chunks = self.split_text(page_text)
            for chunk_text in split_chunks:
                chunk_id = f"{doc_id}_p{page.page_number}_c{global_chunk_idx}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        document_name=document_name,
                        page_number=page.page_number,
                        chunk_index=global_chunk_idx,
                        text=chunk_text,
                    )
                )
                global_chunk_idx += 1

        return all_chunks
