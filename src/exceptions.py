"""Custom application exceptions for StudyBuddy Local."""


class StudyBuddyError(Exception):
    """Base exception for all StudyBuddy Local errors."""
    pass


class OllamaConnectionError(StudyBuddyError):
    """Raised when Ollama local daemon is not running or unreachable."""
    pass


class ModelNotFoundError(StudyBuddyError):
    """Raised when the specified local model is not downloaded in Ollama."""
    pass


class DocumentExtractionError(StudyBuddyError):
    """Raised when text cannot be extracted from a document."""
    pass


class EmptyDocumentError(DocumentExtractionError):
    """Raised when a document contains no extractable text (e.g. scanned image PDF)."""
    pass


class UnsupportedFileTypeError(StudyBuddyError):
    """Raised when an uploaded file extension is not supported."""
    pass


class OversizedFileError(StudyBuddyError):
    """Raised when an uploaded file exceeds the configured size limit."""
    pass


class QuizValidationError(StudyBuddyError):
    """Raised when generated quiz output fails schema validation."""
    pass


class EmbeddingGenerationError(StudyBuddyError):
    """Raised when local embedding generation fails."""
    pass


class VectorStoreError(StudyBuddyError):
    """Raised when local ChromaDB operations fail."""
    pass


class PrivacyViolationError(StudyBuddyError):
    """Raised when a non-local or prohibited network operation is attempted."""
    pass
