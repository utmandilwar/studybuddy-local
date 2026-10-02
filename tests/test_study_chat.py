"""Tests for StudyChat RAG question answering and SummaryGenerator."""

import pytest
from unittest.mock import MagicMock, patch

from src.schemas import AnswerResponse, SourceReference
from src.study_chat import StudyChat
from src.summary_generator import SummaryGenerator


@pytest.fixture
def mock_retriever_with_results():
    """Mock retriever returning sample source excerpts."""
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        SourceReference(
            document_name="operating_systems.pdf",
            page_number=3,
            chunk_index=0,
            excerpt="Virtual memory allows execution of partially loaded processes.",
            score=0.92,
        ),
        SourceReference(
            document_name="operating_systems.pdf",
            page_number=4,
            chunk_index=1,
            excerpt="Demand paging brings pages into memory only when referenced.",
            score=0.88,
        ),
    ]
    return retriever


@pytest.fixture
def mock_retriever_empty():
    """Mock retriever returning no relevant chunks."""
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    return retriever


@pytest.fixture
def mock_llm():
    """Mock LocalLLM returning generated study explanation."""
    llm = MagicMock()
    llm.generate.return_value = (
        "Virtual memory allows a computer to compensate for physical memory shortages "
        "by transferring pages of data from RAM to disk storage. "
        "Sources: operating_systems.pdf (Pages 3, 4)."
    )
    llm.generate_stream.return_value = iter(["Virtual ", "memory ", "explanation."])
    return llm


def test_study_chat_grounded_answer(mock_retriever_with_results, mock_llm):
    """Test standard grounded question answering flow."""
    chat = StudyChat(retriever=mock_retriever_with_results, llm=mock_llm)
    resp = chat.answer_question("What is virtual memory?")

    assert isinstance(resp, AnswerResponse)
    assert resp.has_sufficient_context is True
    assert len(resp.sources) == 2
    assert resp.sources[0].document_name == "operating_systems.pdf"
    assert resp.sources[0].page_number == 3
    assert "Virtual memory allows" in resp.answer
    mock_llm.generate.assert_called_once()


def test_study_chat_insufficient_context(mock_retriever_empty, mock_llm):
    """When no relevant excerpts are found, return honest notice without calling LLM."""
    chat = StudyChat(retriever=mock_retriever_empty, llm=mock_llm)
    resp = chat.answer_question("What is quantum computing entanglement?")

    assert resp.has_sufficient_context is False
    assert len(resp.sources) == 0
    assert "do not contain sufficient information" in resp.answer
    mock_llm.generate.assert_not_called()


def test_study_chat_empty_question():
    """Empty question input raises ValueError."""
    chat = StudyChat()
    with pytest.raises(ValueError):
        chat.answer_question("   ")


def test_study_chat_streaming(mock_retriever_with_results, mock_llm):
    """Test token stream generator for live chat rendering."""
    chat = StudyChat(retriever=mock_retriever_with_results, llm=mock_llm)
    stream, sources, sufficient = chat.answer_question_stream("Explain virtual memory")

    assert sufficient is True
    assert len(sources) == 2
    tokens = list(stream)
    assert "".join(tokens) == "Virtual memory explanation."


def test_summary_generator_modes(mock_retriever_with_results, mock_llm):
    """Test summary generation for short, detailed, and revision modes."""
    summarizer = SummaryGenerator(retriever=mock_retriever_with_results, llm=mock_llm)

    # Short summary
    short_resp = summarizer.generate_summary(mode="short")
    assert short_resp.has_sufficient_context is True
    assert len(short_resp.sources) == 2

    # Detailed summary
    detailed_resp = summarizer.generate_summary(mode="detailed")
    assert detailed_resp.has_sufficient_context is True

    # Revision notes
    rev_resp = summarizer.generate_summary(mode="revision")
    assert rev_resp.has_sufficient_context is True


def test_summary_generator_invalid_mode(mock_retriever_with_results, mock_llm):
    """Invalid summary mode raises ValueError."""
    summarizer = SummaryGenerator(retriever=mock_retriever_with_results, llm=mock_llm)
    with pytest.raises(ValueError):
        summarizer.generate_summary(mode="ultra_fast")
