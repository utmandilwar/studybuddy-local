"""Semantic retrieval of grounded study excerpts for StudyBuddy Local.

Performs local semantic search, document filtering, and similarity thresholding.
"""

from __future__ import annotations

from typing import List, Optional

from config import AppConfig
from src.embeddings import LocalEmbeddingManager
from src.schemas import SourceReference
from src.vector_store import LocalVectorStore


class LocalRetriever:
    """Retrieves relevant study notes using local embeddings and ChromaDB."""

    def __init__(
        self,
        vector_store: Optional[LocalVectorStore] = None,
        embedding_manager: Optional[LocalEmbeddingManager] = None,
    ):
        self.vector_store = vector_store or LocalVectorStore()
        self.embedding_manager = embedding_manager or LocalEmbeddingManager()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_ids: Optional[List[str]] = None,
        relevance_threshold: Optional[float] = None,
    ) -> List[SourceReference]:
        """Retrieve most relevant study excerpts for a query.

        Args:
            query: Question or topic submitted by the student.
            top_k: Max chunks to retrieve.
            doc_ids: Optional list of document IDs to restrict retrieval to.
            relevance_threshold: Minimum cosine similarity score (0.0 to 1.0).

        Returns:
            List of SourceReference instances sorted by relevance.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        k = top_k or AppConfig.TOP_K_CHUNKS
        threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else AppConfig.RELEVANCE_THRESHOLD
        )

        query_emb = self.embedding_manager.generate_query_embedding(clean_query)
        results = self.vector_store.query(
            query_embedding=query_emb,
            top_k=k,
            doc_ids=doc_ids,
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        source_refs: List[SourceReference] = []

        for doc_text, meta, dist in zip(docs, metadatas, distances):
            # For cosine distance: similarity = 1 - distance
            similarity = max(0.0, 1.0 - dist)

            if similarity >= threshold:
                source_refs.append(
                    SourceReference(
                        document_name=meta.get("document_name", "Unknown Document"),
                        page_number=meta.get("page_number"),
                        chunk_index=meta.get("chunk_index", 0),
                        excerpt=doc_text,
                        score=round(float(similarity), 4),
                    )
                )

        return source_refs
