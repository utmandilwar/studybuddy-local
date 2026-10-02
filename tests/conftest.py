"""Shared pytest fixtures for StudyBuddy Local test suite."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture
def sample_study_text() -> str:
    """Sample study text for testing chunking and question answering."""
    return (
        "Operating systems manage computer hardware and software resources. "
        "A process is an instance of a program in execution. Processes require "
        "CPU time, memory, files, and I/O devices to accomplish their tasks. "
        "The operating system scheduler decides which process runs on the CPU. "
        "Virtual memory allows execution of processes that are not completely in memory. "
        "Paging and segmentation are common memory management schemes."
    )


@pytest.fixture
def mock_ollama_tags_response():
    """Mock response for Ollama /api/tags endpoint."""
    return {
        "models": [
            {"name": "qwen2.5:3b", "size": 1900000000},
            {"name": "llama3.2:3b", "size": 2000000000},
        ]
    }


@pytest.fixture
def mock_ollama_generate_response():
    """Mock response for Ollama /api/generate endpoint."""
    return {
        "model": "qwen2.5:3b",
        "response": (
            "An operating system manages hardware resources and schedules processes. "
            "According to the notes, it allocates CPU, memory, and devices. "
            "[Source: notes.txt, Page 1]"
        ),
        "done": True,
    }
