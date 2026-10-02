"""Tests for privacy guarantees, local endpoints, and log sanitization."""

import pytest
from config import AppConfig
from src.exceptions import PrivacyViolationError
from src.privacy import (
    assert_local_configuration,
    get_privacy_badge,
    get_privacy_disclosure,
    is_local_endpoint,
    sanitize_for_log,
)


def test_is_local_endpoint_valid():
    """Localhost, 127.0.0.1, and loopback must be recognized as local."""
    assert is_local_endpoint("http://localhost:11434")
    assert is_local_endpoint("http://127.0.0.1:11434")
    assert is_local_endpoint("http://0.0.0.0:8501")


def test_is_local_endpoint_blocks_external():
    """External internet domains must be rejected."""
    assert not is_local_endpoint("https://api.openai.com/v1")
    assert not is_local_endpoint("https://generativelanguage.googleapis.com")
    assert not is_local_endpoint("https://api.anthropic.com")
    assert not is_local_endpoint("https://my-cloud-ollama.example.com")


def test_assert_local_configuration_default():
    """Default application config must pass local privacy assertion."""
    # Should not raise
    assert_local_configuration()


def test_assert_local_configuration_raises_on_cloud(monkeypatch):
    """If Ollama URL is configured to an external cloud URL, raise PrivacyViolationError."""
    monkeypatch.setattr(AppConfig, "OLLAMA_BASE_URL", "https://api.openai.com/v1")
    with pytest.raises(PrivacyViolationError) as exc_info:
        assert_local_configuration()
    assert "Prohibited external AI endpoint" in str(exc_info.value) or "Non-local" in str(exc_info.value)


def test_get_privacy_badge_local():
    """Verify that the privacy badge returns the exact required text."""
    badge, verified = get_privacy_badge()
    assert verified is True
    assert badge == "Local AI Mode — Your study content is processed on this device."


def test_get_privacy_disclosure_contains_required_sections():
    """Check that privacy disclosure details local processing, zero telemetry, and limits."""
    disclosure = get_privacy_disclosure()
    assert "local_execution" in disclosure
    assert "local_embeddings" in disclosure
    assert "local_storage" in disclosure
    assert "zero_telemetry" in disclosure
    assert "transparency_boundary" in disclosure


def test_sanitize_for_log():
    """Full study notes or long prompts must never be dumped raw into logs."""
    sensitive_text = "Alice's private exam answer sheet with secret questions and notes."
    sanitized = sanitize_for_log(sensitive_text, max_chars=15)
    assert len(sanitized) < len(sensitive_text) + 30
    assert "preview: Alice's private..." in sanitized
    assert f"{len(sensitive_text)} chars" in sanitized
    assert sensitive_text not in sanitized


def test_no_prohibited_dependencies_in_requirements():
    """Ensure requirements.txt does not contain cloud AI SDKs."""
    req_path = AppConfig.PROJECT_ROOT / "requirements.txt"
    assert req_path.exists()
    content = req_path.read_text(encoding="utf-8").lower()

    prohibited = ["openai", "google-generativeai", "anthropic", "cohere", "pinecone"]
    for package in prohibited:
        assert package not in content, f"Prohibited cloud package '{package}' found in requirements.txt"
