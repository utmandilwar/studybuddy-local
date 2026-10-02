"""Privacy enforcement, validation, and transparency utilities for StudyBuddy Local.

Ensures no cloud endpoints are configured, guards against external transmissions,
and provides auditable privacy status for the user interface.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

from config import AppConfig
from src.exceptions import PrivacyViolationError

# Disallowed domain substrings (hosted AI, cloud analytics, tracking)
BLOCKED_HOST_PATTERNS = [
    r"openai\.com",
    r"anthropic\.com",
    r"googleapis\.com",
    r"azure\.com",
    r"cohere\.ai",
    r"pinecone\.io",
    r"weaviate\.cloud",
    r"qdrant\.io",
    r"segment\.io",
    r"posthog\.com",
    r"sentry\.io",
    r"datadoghq\.com",
]


def is_local_endpoint(url: str) -> bool:
    """Verify that an endpoint URL resolves strictly to localhost or loopback."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        allowed_hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
        return hostname.lower() in allowed_hosts
    except Exception:
        return False


def assert_local_configuration(endpoint_url: Optional[str] = None) -> None:
    """Assert that all core service endpoints conform strictly to local-only operation.

    Args:
        endpoint_url: Optional specific URL to check. If None, checks AppConfig.OLLAMA_BASE_URL.

    Raises:
        PrivacyViolationError: If any configured endpoint points to an external or cloud service.
    """
    ollama_url = endpoint_url or AppConfig.OLLAMA_BASE_URL
    if not is_local_endpoint(ollama_url):
        raise PrivacyViolationError(
            f"Non-local Ollama endpoint detected: {ollama_url}. "
            "StudyBuddy Local must bind strictly to local addresses (localhost / 127.0.0.1)."
        )

    for pattern in BLOCKED_HOST_PATTERNS:
        if re.search(pattern, ollama_url, re.IGNORECASE):
            raise PrivacyViolationError(
                f"Prohibited external AI endpoint detected matching {pattern}."
            )


def get_privacy_badge() -> Tuple[str, bool]:
    """Return the privacy badge text and whether the system is fully local.

    Returns:
        (badge_text, is_local_verified)
    """
    try:
        assert_local_configuration()
        return ("Local AI Mode — Your study content is processed on this device.", True)
    except PrivacyViolationError:
        return ("Warning: Non-Local Endpoint Detected", False)


def get_privacy_disclosure() -> Dict[str, str]:
    """Return transparent, honest privacy explanations for users."""
    return {
        "local_execution": (
            "All text generation and question answering run on your device via Ollama. "
            "Your prompts and study notes are not sent to any hosted AI service."
        ),
        "local_embeddings": (
            "Document embeddings are computed locally using Sentence Transformers. "
            "No vectors or raw texts are transmitted to external embedding APIs."
        ),
        "local_storage": (
            "Uploaded documents and the ChromaDB vector database are stored directly on "
            f"your local disk in the application directory ({AppConfig.DATA_DIR})."
        ),
        "zero_telemetry": (
            "StudyBuddy Local contains no usage tracking, Google Analytics, or external "
            "telemetry. Streamlit usage stats collection is explicitly disabled."
        ),
        "transparency_boundary": (
            "While StudyBuddy Local strictly isolates its application data flow, system-level "
            "OS networking, system updates, and third-party software fall outside the "
            "application's control. Complete isolation can be tested by running the application "
            "with your network connection disconnected."
        ),
    }


def sanitize_for_log(text: str, max_chars: int = 40) -> str:
    """Helper to prevent logging full document contents or personal study material.

    Args:
        text: Input string.
        max_chars: Max allowed characters before truncation.

    Returns:
        Safe preview string showing only character length.
    """
    if not text:
        return "[empty]"
    length = len(text)
    return f"[Content: {length} chars, preview: {text[:min(length, max_chars)].strip()}...]"
