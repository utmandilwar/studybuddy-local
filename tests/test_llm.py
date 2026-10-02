"""Tests for local LLM communication, connection handling, and JSON parsing."""

import json
import pytest
import httpx
from unittest.mock import patch, MagicMock

from config import AppConfig
from src.exceptions import ModelNotFoundError, OllamaConnectionError, PrivacyViolationError
from src.llm import LocalLLM


def test_llm_rejects_external_base_url():
    """LocalLLM must immediately reject non-localhost base URLs."""
    with pytest.raises(PrivacyViolationError):
        LocalLLM(base_url="https://api.openai.com/v1")


def test_llm_check_connection_offline_handled():
    """When Ollama is not running, check_connection returns graceful tuple, not uncaught crash."""
    llm = LocalLLM(base_url="http://127.0.0.1:54321")  # closed port
    connected, message, models = llm.check_connection()
    assert connected is False
    assert "Cannot connect to Ollama" in message or "error" in message.lower()
    assert models == []


def test_llm_generate_success(mock_ollama_generate_response):
    """Test successful generation using mocked local Ollama response."""
    llm = LocalLLM()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_ollama_generate_response

    with patch("httpx.Client.post", return_value=mock_resp):
        ans = llm.generate("Explain operating systems.")
        assert "An operating system manages hardware" in ans


def test_llm_generate_model_not_found():
    """Test that HTTP 404 from Ollama translates into ModelNotFoundError."""
    llm = LocalLLM()

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "model 'qwen2.5:3b' not found"

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(ModelNotFoundError) as exc:
            llm.generate("Explain something.")
        assert "ollama pull" in str(exc.value)


def test_llm_generate_json_parsing():
    """Test JSON response parsing including fenced markdown extraction."""
    llm = LocalLLM()

    expected_payload = {
        "questions": [
            {
                "question": "What does an OS manage?",
                "options": {"A": "Hardware", "B": "Compilers", "C": "Cables", "D": "Screens"},
                "correct_answer": "A",
                "explanation": "OS manages hardware.",
            }
        ]
    }

    # Case 1: Plain JSON
    with patch.object(llm, "generate", return_value=json.dumps(expected_payload)):
        res = llm.generate_json("Generate quiz")
        assert res["questions"][0]["correct_answer"] == "A"

    # Case 2: Markdown fenced JSON
    fenced_output = f"Here is the quiz:\n```json\n{json.dumps(expected_payload)}\n```\nGood luck!"
    with patch.object(llm, "generate", return_value=fenced_output):
        res = llm.generate_json("Generate quiz")
        assert res["questions"][0]["correct_answer"] == "A"

    # Case 3: Invalid JSON raises ValueError
    with patch.object(llm, "generate", return_value="Sorry I cannot do this."):
        with pytest.raises(ValueError):
            llm.generate_json("Generate quiz")
