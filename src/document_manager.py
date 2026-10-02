"""Document library management and persistence for StudyBuddy Local.

Tracks indexed files, handles deduplication, and cleans up vector store records.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import AppConfig
from src.chunker import TextChunker
from src.document_loader import DocumentLoader
from src.embeddings import LocalEmbeddingManager
from src.exceptions import StudyBuddyError
from src.schemas import DocumentMetadata
from src.vector_store import LocalVectorStore


class DocumentManager:
    """Coordinates document ingestion, metadata recording, and index cleanup."""

    def __init__(
        self,
        metadata_file: Optional[Path] = None,
        vector_store: Optional[LocalVectorStore] = None,
        embedding_manager: Optional[LocalEmbeddingManager] = None,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[TextChunker] = None,
    ):
        self.metadata_file = metadata_file or AppConfig.DOCS_METADATA_PATH
        self.vector_store = vector_store or LocalVectorStore()
        self.embedding_manager = embedding_manager or LocalEmbeddingManager()
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or TextChunker()

    def _generate_doc_id(self, filename: str, file_bytes: bytes) -> str:
        """Create a deterministic doc_id based on filename and content hash."""
        sha = hashlib.sha256(file_bytes).hexdigest()[:12]
        safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")
        return f"{safe_name}_{sha}"

    def _load_metadata_map(self) -> Dict[str, DocumentMetadata]:
        """Load stored document metadata from local JSON file."""
        if not self.metadata_file.exists():
            return {}
        try:
            raw = json.loads(self.metadata_file.read_text(encoding="utf-8"))
            return {k: DocumentMetadata(**v) for k, v in raw.items()}
        except Exception:
            return {}

    def _save_metadata_map(self, data: Dict[str, DocumentMetadata]) -> None:
        """Persist metadata dictionary to local JSON file."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        serialized = {k: v.model_dump() for k, v in data.items()}
        self.metadata_file.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def list_documents(self) -> List[DocumentMetadata]:
        """List all indexed study documents."""
        docs = list(self._load_metadata_map().values())
        docs.sort(key=lambda d: d.uploaded_at, reverse=True)
        return docs

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Find metadata for a single document by doc_id."""
        return self._load_metadata_map().get(doc_id)

    def process_and_index_document(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[DocumentMetadata, int]:
        """Ingest, extract, chunk, embed, and index an uploaded document.

        Args:
            file_bytes: Raw binary content of the file.
            filename: Uploaded file name.

        Returns:
            Tuple of (DocumentMetadata, indexed_chunks_count)
        """
        doc_id = self._generate_doc_id(filename, file_bytes)

        # 1. Extract pages safely
        pages = self.loader.load_from_bytes(file_bytes, filename)
        page_count = len(pages)

        # 2. Chunk text preserving page metadata
        chunks = self.chunker.chunk_pages(pages, doc_id=doc_id, document_name=filename)
        if not chunks:
            raise StudyBuddyError(f"No extractable chunks found in '{filename}'.")

        # 3. Clean up any previous chunks if re-uploading the same file
        self.vector_store.delete_document(doc_id)

        # 4. Generate embeddings locally
        texts = [c.text for c in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        # 5. Store in persistent local ChromaDB
        indexed_count = self.vector_store.add_chunks(chunks, embeddings)

        # 6. Save metadata record
        meta = DocumentMetadata(
            doc_id=doc_id,
            name=filename,
            size_bytes=len(file_bytes),
            page_count=page_count,
            chunk_count=indexed_count,
            uploaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status="indexed",
        )

        metadata_map = self._load_metadata_map()
        metadata_map[doc_id] = meta
        self._save_metadata_map(metadata_map)

        return meta, indexed_count

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and its associated vectors.

        Args:
            doc_id: Identifier of the document to delete.

        Returns:
            True if deleted, False if document was not found.
        """
        metadata_map = self._load_metadata_map()
        if doc_id not in metadata_map:
            return False

        # Remove vector embeddings from ChromaDB
        self.vector_store.delete_document(doc_id)

        # Remove from local metadata registry
        del metadata_map[doc_id]
        self._save_metadata_map(metadata_map)
        return True
