"""Pydantic data schemas for StudyBuddy Local."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """Represents an extracted and chunked piece of study text."""

    chunk_id: str
    doc_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_index: int
    text: str


class SourceReference(BaseModel):
    """Source citation supporting an answer or quiz question."""

    document_name: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    excerpt: str
    score: Optional[float] = None


class DocumentMetadata(BaseModel):
    """Metadata recorded for an indexed document in the local library."""

    doc_id: str
    name: str
    size_bytes: int
    page_count: int = 1
    chunk_count: int = 0
    uploaded_at: str
    status: str = "indexed"
    error_message: Optional[str] = None


class QuizQuestion(BaseModel):
    """A single multiple choice question with grounded explanation."""

    question: str = Field(..., description="The question text grounded in notes")
    options: Dict[str, str] = Field(
        ..., description="Four options keyed by 'A', 'B', 'C', 'D'"
    )
    correct_answer: str = Field(
        ..., description="The key of the correct option ('A', 'B', 'C', or 'D')"
    )
    explanation: str = Field(
        ..., description="Explanation citing study content on why the answer is correct"
    )
    source_document: Optional[str] = None
    source_page: Optional[int] = None

    @field_validator("options")
    @classmethod
    def validate_options(cls, v: Dict[str, str]) -> Dict[str, str]:
        expected_keys = {"A", "B", "C", "D"}
        if set(v.keys()) != expected_keys:
            raise ValueError(f"Options must contain exactly keys A, B, C, D; got {list(v.keys())}")
        values = [val.strip().lower() for val in v.values() if val.strip()]
        if len(set(values)) != 4:
            raise ValueError("All 4 options must be distinct and non-empty")
        return v

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer(cls, v: str, info) -> str:
        clean = v.strip().upper()
        if clean not in {"A", "B", "C", "D"}:
            raise ValueError(f"correct_answer must be one of A, B, C, D; got {v}")
        return clean


class GeneratedQuiz(BaseModel):
    """A set of validated practice quiz questions."""

    questions: List[QuizQuestion]

    @field_validator("questions")
    @classmethod
    def validate_questions_list(cls, v: List[QuizQuestion]) -> List[QuizQuestion]:
        if not v:
            raise ValueError("Quiz must contain at least 1 question")
        return v


class AnswerResponse(BaseModel):
    """The generated answer accompanied by grounded source citations."""

    answer: str
    sources: List[SourceReference] = Field(default_factory=list)
    has_sufficient_context: bool = True
