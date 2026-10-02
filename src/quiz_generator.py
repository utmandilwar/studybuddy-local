"""Practice quiz generation and scoring engine for StudyBuddy Local.

Generates multiple-choice questions grounded in student notes with schema validation,
single-retry error resilience, and answer verification.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from config import AppConfig
from src.exceptions import QuizValidationError, StudyBuddyError
from src.llm import LocalLLM
from src.prompts import QUIZ_PROMPT_TEMPLATE, STUDY_TUTOR_SYSTEM_PROMPT
from src.retriever import LocalRetriever
from src.schemas import GeneratedQuiz, QuizQuestion, SourceReference


class QuizSession:
    """Manages the interactive state of an ongoing quiz session."""

    def __init__(self, quiz: GeneratedQuiz):
        self.quiz = quiz
        self.user_answers: Dict[int, str] = {}
        self.submitted_questions: set[int] = set()

    @property
    def total_questions(self) -> int:
        return len(self.quiz.questions)

    def submit_answer(self, question_index: int, selected_option: str) -> Dict[str, Any]:
        """Submit an answer for a specific question index.

        Args:
            question_index: 0-based index of the question.
            selected_option: One of 'A', 'B', 'C', 'D'.

        Returns:
            Dict containing correctness, correct answer, and explanation.
        """
        if question_index < 0 or question_index >= self.total_questions:
            raise IndexError("Question index out of bounds")

        clean_opt = selected_option.strip().upper()
        q = self.quiz.questions[question_index]
        is_correct = clean_opt == q.correct_answer

        self.user_answers[question_index] = clean_opt
        self.submitted_questions.add(question_index)

        return {
            "is_correct": is_correct,
            "user_choice": clean_opt,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "source_document": q.source_document,
            "source_page": q.source_page,
        }

    def is_submitted(self, question_index: int) -> bool:
        return question_index in self.submitted_questions

    def get_user_answer(self, question_index: int) -> Optional[str]:
        return self.user_answers.get(question_index)

    def is_complete(self) -> bool:
        return len(self.submitted_questions) == self.total_questions

    def calculate_score(self) -> Dict[str, Any]:
        """Calculate total score and percentage for submitted answers."""
        if self.total_questions == 0:
            return {"correct": 0, "total": 0, "percentage": 0.0}

        correct_count = 0
        for idx, q in enumerate(self.quiz.questions):
            if self.user_answers.get(idx) == q.correct_answer:
                correct_count += 1

        percentage = round((correct_count / self.total_questions) * 100, 1)
        return {
            "correct": correct_count,
            "total": self.total_questions,
            "percentage": percentage,
        }

    def reset(self) -> None:
        """Clear user answers and submissions to restart the quiz."""
        self.user_answers.clear()
        self.submitted_questions.clear()


class QuizGenerator:
    """Generates schema-validated quizzes using local LLM and local notes."""

    def __init__(
        self,
        retriever: Optional[LocalRetriever] = None,
        llm: Optional[LocalLLM] = None,
    ):
        self.retriever = retriever or LocalRetriever()
        self.llm = llm or LocalLLM()

    def _gather_study_passages(
        self,
        doc_ids: Optional[List[str]] = None,
        topic_focus: Optional[str] = None,
        max_chunks: int = 6,
    ) -> List[SourceReference]:
        """Collect source passages to ground quiz questions."""
        query = (
            topic_focus.strip()
            if topic_focus and topic_focus.strip()
            else "important concepts definitions facts principles"
        )
        return self.retriever.retrieve(
            query=query,
            top_k=max_chunks,
            doc_ids=doc_ids,
            relevance_threshold=0.05,
        )

    def generate_quiz(
        self,
        doc_ids: Optional[List[str]] = None,
        num_questions: int = 5,
        difficulty: str = "Medium",
        topic_focus: Optional[str] = None,
    ) -> GeneratedQuiz:
        """Generate a practice quiz with schema validation and automatic retry.

        Args:
            doc_ids: Optional list of document IDs to query.
            num_questions: Number of questions (default 5).
            difficulty: 'Easy', 'Medium', or 'Hard'.
            topic_focus: Optional concept/chapter to target.

        Returns:
            Validated GeneratedQuiz object.

        Raises:
            StudyBuddyError: If no study material exists to quiz on.
            QuizValidationError: If LLM produces invalid schema after retry.
        """
        sources = self._gather_study_passages(
            doc_ids=doc_ids, topic_focus=topic_focus, max_chunks=6
        )

        if not sources:
            raise StudyBuddyError(
                "Cannot generate quiz: No relevant indexed study notes were found. "
                "Please upload lecture notes or textbooks first."
            )

        # Build study excerpts context
        context_parts = []
        for idx, src in enumerate(sources, 1):
            page_info = f", Page {src.page_number}" if src.page_number else ""
            context_parts.append(f"[Excerpt {idx}: {src.document_name}{page_info}]\n{src.excerpt}")
        context_str = "\n\n".join(context_parts)

        default_doc = sources[0].document_name

        prompt = QUIZ_PROMPT_TEMPLATE.format(
            num_questions=num_questions,
            difficulty=difficulty,
            context=context_str,
            default_doc_name=default_doc,
        )

        # Attempt 1
        try:
            raw_json = self.llm.generate_json(
                prompt=prompt, system_prompt=STUDY_TUTOR_SYSTEM_PROMPT
            )
            return self._validate_and_sanitize(raw_json, default_doc)
        except Exception as first_error:
            # Attempt 2: Targeted Retry with corrective prompt
            retry_prompt = (
                f"{prompt}\n\n"
                f"NOTICE: Your previous output was invalid due to: {str(first_error)}.\n"
                "Please fix this and output ONLY valid JSON adhering strictly to the schema."
            )
            try:
                raw_json_retry = self.llm.generate_json(
                    prompt=retry_prompt, system_prompt=STUDY_TUTOR_SYSTEM_PROMPT
                )
                return self._validate_and_sanitize(raw_json_retry, default_doc)
            except Exception as second_error:
                raise QuizValidationError(
                    f"Failed to generate a valid quiz after retry: {str(second_error)}"
                ) from second_error

    def _validate_and_sanitize(
        self, raw_data: Any, fallback_doc_name: str
    ) -> GeneratedQuiz:
        """Validate raw dictionary against GeneratedQuiz Pydantic schema."""
        if not isinstance(raw_data, dict):
            raise ValueError(f"Expected JSON object, got {type(raw_data).__name__}")

        questions_list = raw_data.get("questions")
        if not isinstance(questions_list, list) or not questions_list:
            raise ValueError("JSON must contain a non-empty 'questions' list.")

        sanitized_questions = []
        for q in questions_list:
            if not isinstance(q, dict):
                continue
            # Ensure document fallback
            if not q.get("source_document"):
                q["source_document"] = fallback_doc_name
            sanitized_questions.append(q)

        validated = GeneratedQuiz(questions=sanitized_questions)
        return validated
