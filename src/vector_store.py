"""Local persistent vector storage using ChromaDB for StudyBuddy Local.

All indexes are persisted to local disk (data/chroma).
Zero remote server calls or cloud telemetry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config import AppConfig
from src.exceptions import VectorStoreError
from src.schemas import DocumentChunk


class LocalVectorStore:
    """Manages persistent ChromaDB vector storage on local disk."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = persist_directory or AppConfig.CHROMA_PATH
        self.collection_name = collection_name or AppConfig.COLLECTION_NAME
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize persistent local Chroma client with telemetry disabled
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False, is_persistent=True),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB at {self.persist_dir}: {str(e)}") from e

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Add or update chunks in the persistent local index.

        Args:
            chunks: List of DocumentChunk schemas.
            embeddings: Computed vector embeddings for each chunk.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count must match embeddings count")

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "document_name": c.document_name,
                "page_number": c.page_number if c.page_number is not None else 1,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        try:
            # Use upsert to handle re-uploads and updates cleanly without duplicates
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            return len(chunks)
        except Exception as e:
            raise VectorStoreError(f"Failed to add chunks to ChromaDB: {str(e)}") from e

    def query(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform semantic search against local chunks.

        Args:
            query_embedding: Dense embedding vector of the search query.
            top_k: Number of relevant results to retrieve.
            doc_ids: Optional list of document IDs to restrict search to.

        Returns:
            Chroma query result dictionary.
        """
        k = top_k or AppConfig.TOP_K_CHUNKS
        total_items = self.collection.count()
        if total_items == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        effective_k = min(k, total_items)
        where_filter: Optional[Dict[str, Any]] = None

        if doc_ids:
            if len(doc_ids) == 1:
                where_filter = {"doc_id": doc_ids[0]}
            else:
                where_filter = {"doc_id": {"$in": doc_ids}}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            raise VectorStoreError(f"ChromaDB query failed: {str(e)}") from e

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks associated with a specific document from the index.

        Args:
            doc_id: Unique document identifier to remove.
        """
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            raise VectorStoreError(f"Failed to delete document '{doc_id}' from ChromaDB: {str(e)}") from e

    def get_chunk_count_for_document(self, doc_id: str) -> int:
        """Count the number of indexed chunks belonging to a document."""
        try:
            res = self.collection.get(where={"doc_id": doc_id})
            return len(res.get("ids", []))
        except Exception:
            return 0

    def get_total_chunk_count(self) -> int:
        """Return total indexed chunks across all study documents."""
        try:
            return self.collection.count()
        except Exception:
            return 0
