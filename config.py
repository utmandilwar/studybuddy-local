"""Configuration management for StudyBuddy Local.

All configuration defaults are strictly local-first with zero cloud API keys,
zero external telemetry, and persistent local storage paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
DOCS_METADATA_FILE = DATA_DIR / "documents.json"

# Ensure runtime directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
CHROMA_DIR.mkdir(exist_ok=True, parents=True)


class AppConfig:
    """Application settings for local inference, RAG, and privacy."""

    # Application details
    APP_NAME: str = "StudyBuddy Local"
    APP_VERSION: str = "1.0.0"
    TAGLINE: str = "Your private AI study partner. Your notes stay yours."

    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = DATA_DIR
    CHROMA_PATH: Path = CHROMA_DIR
    DOCS_METADATA_PATH: Path = DOCS_METADATA_FILE

    # Ollama Local LLM Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    DEFAULT_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    FALLBACK_MODELS: list[str] = ["llama3.2:3b", "phi3:mini", "mistral:7b"]
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))

    # Local Embeddings Settings
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Document Chunking Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval Settings
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "5"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.20"))

    # Upload Limits & Constraints
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt"]

    # Storage Paths
    CHROMA_PATH: Path = CHROMA_DIR
    COLLECTION_NAME: str = "studybuddy_notes"

    # Context & Generation Parameters
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 1500

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return the non-sensitive configuration dictionary."""
        return {
            "app_name": cls.APP_NAME,
            "version": cls.APP_VERSION,
            "ollama_base_url": cls.OLLAMA_BASE_URL,
            "default_model": cls.DEFAULT_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL_NAME,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "top_k": cls.TOP_K_CHUNKS,
            "relevance_threshold": cls.RELEVANCE_THRESHOLD,
            "max_upload_size_mb": cls.MAX_UPLOAD_SIZE_MB,
            "chroma_dir": str(cls.CHROMA_PATH),
        }
