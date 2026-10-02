"""Tests for QuizGenerator and QuizSession validation, scoring, and error recovery."""

import pytest
from unittest.mock import MagicMock, patch

from src.exceptions import QuizValidationError, StudyBuddyError
from src.quiz_generator import QuizGenerator, QuizSession
from src.schemas import GeneratedQuiz, QuizQuestion, SourceReference


@pytest.fixture
def sample_valid_quiz_payload():
    """Valid raw quiz dictionary conforming to schema."""
    return {
        "questions": [
            {
                "question": "What is the primary role of an operating system kernel?",
                "options": {
                    "A": "Manage system hardware and execution mode",
                    "B": "Design graphic user interfaces",
                    "C": "Compile all python scripts",
                    "D": "Assemble physical network cables",
                },
                "correct_answer": "A",
                "explanation": "The kernel is the core component that manages hardware resources directly.",
                "source_document": "os_intro.pdf",
                "source_page": 2,
            },
            {
                "question": "What happens during a page fault?",
                "options": {
                    "A": "CPU shuts down immediately",
                    "B": "Operating system fetches requested page from secondary storage into RAM",
                    "C": "RAM memory is wiped clean",
                    "D": "Monitor screen flickers off",
                },
                "correct_answer": "B",
                "explanation": "A page fault occurs when a referenced memory page is not in physical RAM.",
                "source_document": "os_intro.pdf",
                "source_page": 5,
            },
        ]
    }


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        SourceReference(
            document_name="os_intro.pdf",
            page_number=2,
            chunk_index=0,
            excerpt="Kernel manages hardware and CPU execution modes.",
            score=0.9,
        )
    ]
    return retriever


def test_valid_quiz_accepted(sample_valid_quiz_payload, mock_retriever):
    """A schema-compliant quiz payload parses successfully."""
    mock_llm = MagicMock()
    mock_llm.generate_json.return_value = sample_valid_quiz_payload

    generator = QuizGenerator(retriever=mock_retriever, llm=mock_llm)
    quiz = generator.generate_quiz(num_questions=2)

    assert isinstance(quiz, GeneratedQuiz)
    assert len(quiz.questions) == 2
    assert quiz.questions[0].correct_answer == "A"
    assert quiz.questions[1].correct_answer == "B"


def test_quiz_missing_required_fields_rejected():
    """Questions with missing explanation or options raise ValidationError."""
    invalid_data = {
        "question": "What is RAM?",
        "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
        "correct_answer": "A",
        # missing 'explanation'
    }
    with pytest.raises(Exception):
        QuizQuestion(**invalid_data)


def test_quiz_invalid_correct_answer_reference():
    """Correct answer not in [A, B, C, D] must be rejected."""
    invalid_data = {
        "question": "What is RAM?",
        "options": {"A": "Opt1", "B": "Opt2", "C": "Opt3", "D": "Opt4"},
        "correct_answer": "E",  # Invalid key
        "explanation": "Explanation here.",
    }
    with pytest.raises(Exception):
        QuizQuestion(**invalid_data)


def test_quiz_duplicate_options_rejected():
    """Duplicate option values must be rejected."""
    invalid_data = {
        "question": "What is an OS?",
        "options": {
            "A": "Hardware manager",
            "B": "Hardware manager",  # duplicate of A
            "C": "Compiler",
            "D": "Browser",
        },
        "correct_answer": "A",
        "explanation": "Must be distinct.",
    }
    with pytest.raises(ValueError):
        QuizQuestion(**invalid_data)


def test_quiz_retry_on_first_failure(sample_valid_quiz_payload, mock_retriever):
    """When the first generation fails, generator retries once and succeeds."""
    mock_llm = MagicMock()
    # First call fails, second call succeeds
    mock_llm.generate_json.side_effect = [
        ValueError("Malformed JSON response from model"),
        sample_valid_quiz_payload,
    ]

    generator = QuizGenerator(retriever=mock_retriever, llm=mock_llm)
    quiz = generator.generate_quiz(num_questions=2)

    assert len(quiz.questions) == 2
    assert mock_llm.generate_json.call_count == 2


def test_quiz_fails_after_retry(mock_retriever):
    """When both attempts fail, raise QuizValidationError."""
    mock_llm = MagicMock()
    mock_llm.generate_json.side_effect = [
        ValueError("First failure"),
        ValueError("Second failure"),
    ]

    generator = QuizGenerator(retriever=mock_retriever, llm=mock_llm)
    with pytest.raises(QuizValidationError):
        generator.generate_quiz(num_questions=2)


def test_quiz_no_study_materials_error():
    """Generating quiz without uploaded notes raises StudyBuddyError."""
    empty_retriever = MagicMock()
    empty_retriever.retrieve.return_value = []
    mock_llm = MagicMock()

    generator = QuizGenerator(retriever=empty_retriever, llm=mock_llm)
    with pytest.raises(StudyBuddyError):
        generator.generate_quiz()


def test_quiz_session_state_and_scoring(sample_valid_quiz_payload):
    """Test interactive answer submission, correctness check, and score calculation."""
    quiz = GeneratedQuiz(**sample_valid_quiz_payload)
    session = QuizSession(quiz)

    assert session.total_questions == 2
    assert session.is_complete() is False

    # Question 0: Correct answer is 'A'
    res0 = session.submit_answer(0, "A")
    assert res0["is_correct"] is True
    assert res0["correct_answer"] == "A"
    assert "explanation" in res0
    assert session.is_submitted(0) is True

    # Question 1: Submit wrong answer 'C' (correct is 'B')
    res1 = session.submit_answer(1, "C")
    assert res1["is_correct"] is False
    assert res1["correct_answer"] == "B"

    # Completed
    assert session.is_complete() is True
    score = session.calculate_score()
    assert score["correct"] == 1
    assert score["total"] == 2
    assert score["percentage"] == 50.0

    # Reset
    session.reset()
    assert session.is_complete() is False
    assert session.get_user_answer(0) is None
