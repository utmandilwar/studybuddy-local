"""Document loading and safe text extraction for StudyBuddy Local.

Supports PDF (via PyMuPDF) and plain text (.txt) files.
Extracts text with page tracking, handles scanned/empty documents,
and enforces file size limits.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from config import AppConfig
from src.exceptions import (
    DocumentExtractionError,
    EmptyDocumentError,
    OversizedFileError,
    UnsupportedFileTypeError,
)


class PageContent:
    """Container for text extracted from a specific document page."""

    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = text


class DocumentLoader:
    """Safely extracts text and page metadata from PDF and TXT files."""

    def __init__(self, max_size_mb: Optional[int] = None):
        self.max_size_bytes = (max_size_mb or AppConfig.MAX_UPLOAD_SIZE_MB) * 1024 * 1024

    def validate_file(self, filename: str, file_size: int) -> None:
        """Validate filename extension and file size.

        Raises:
            UnsupportedFileTypeError: If the extension is not .pdf or .txt.
            OversizedFileError: If file size exceeds maximum allowed.
        """
        suffix = Path(filename).suffix.lower()
        if suffix not in AppConfig.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{suffix}'. StudyBuddy Local supports: "
                + ", ".join(AppConfig.SUPPORTED_EXTENSIONS)
            )

        if file_size > self.max_size_bytes:
            limit_mb = self.max_size_bytes // (1024 * 1024)
            size_mb = file_size / (1024 * 1024)
            raise OversizedFileError(
                f"File '{filename}' ({size_mb:.1f} MB) exceeds maximum upload limit of {limit_mb} MB."
            )

    def load_from_bytes(
        self, file_bytes: bytes, filename: str
    ) -> List[PageContent]:
        """Extract pages from in-memory file bytes.

        Args:
            file_bytes: Raw binary content of the file.
            filename: Original uploaded filename.

        Returns:
            List of PageContent objects.
        """
        self.validate_file(filename, len(file_bytes))
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(file_bytes, filename)
        elif suffix == ".txt":
            return self._extract_txt(file_bytes, filename)
        else:
            raise UnsupportedFileTypeError(f"Unsupported extension: {suffix}")

    def load_from_path(self, file_path: Union[str, Path]) -> List[PageContent]:
        """Extract pages from a local filesystem file."""
        path = Path(file_path)
        if not path.exists():
            raise DocumentExtractionError(f"File not found: {path}")

        file_bytes = path.read_bytes()
        return self.load_from_bytes(file_bytes, path.name)

    def _extract_pdf(self, file_bytes: bytes, filename: str) -> List[PageContent]:
        """Extract text from PDF pages using PyMuPDF."""
        if fitz is None:
            raise DocumentExtractionError(
                "PyMuPDF (fitz) is not installed. Please install 'pymupdf'."
            )

        pages: List[PageContent] = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if doc.page_count == 0:
                raise EmptyDocumentError(f"PDF '{filename}' contains 0 pages.")

            total_extracted_chars = 0
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                text = page.get_text("text").strip()
                pages.append(PageContent(page_number=page_idx + 1, text=text))
                total_extracted_chars += len(text)

            doc.close()

            if total_extracted_chars == 0:
                raise EmptyDocumentError(
                    f"No extractable text found in '{filename}'. "
                    "This appears to be a scanned document or image-only PDF. "
                    "(Note: OCR is not supported in this offline version.)"
                )

            return pages
        except EmptyDocumentError:
            raise
        except Exception as e:
            raise DocumentExtractionError(
                f"Failed to process PDF '{filename}': {str(e)}"
            ) from e

    def _extract_txt(self, file_bytes: bytes, filename: str) -> List[PageContent]:
        """Extract plain text from TXT bytes, trying UTF-8 then Latin-1."""
        text = ""
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = file_bytes.decode(encoding).strip()
                break
            except UnicodeDecodeError:
                continue

        if not text:
            raise EmptyDocumentError(
                f"Text file '{filename}' is empty or contains no readable characters."
            )

        return [PageContent(page_number=1, text=text)]
