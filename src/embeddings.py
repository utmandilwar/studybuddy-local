"""Local text embedding generation using Sentence Transformers.

Strictly local-first: All vectors are computed on-device. No cloud APIs.
"""

from __future__ import annotations

from typing import List, Optional

from config import AppConfig
from src.exceptions import EmbeddingGenerationError
from src.privacy import assert_local_configuration


class LocalEmbeddingManager:
    """Manages the local sentence-transformers model for generating embeddings."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or AppConfig.EMBEDDING_MODEL_NAME
        self._model = None
        # Strict local privacy check
        assert_local_configuration()

    @property
    def model(self):
        """Lazy loader for the local sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                # Load model locally; will use cached weights in huggingface cache if available
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise EmbeddingGenerationError(
                    f"Failed to load local embedding model '{self.model_name}': {str(e)}"
                ) from e
        return self._model

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 16
    ) -> List[List[float]]:
        """Generate normalized embedding vectors for a list of texts in small batches.

        Args:
            texts: List of text strings to embed.
            batch_size: Processing batch size optimized for laptop RAM.

        Returns:
            List of float lists representing dense vectors.
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingGenerationError(
                f"Error generating local embeddings: {str(e)}"
            ) from e

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate a single embedding vector for a student query."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        res = self.generate_embeddings([query.strip()], batch_size=1)
        return res[0]
