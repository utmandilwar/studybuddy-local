"""Study chat coordination and RAG question answering pipeline for StudyBuddy Local.

Combines local retrieval, prompt injection defense, and local LLM generation.
"""

from __future__ import annotations

from typing import Generator, List, Optional

from config import AppConfig
from src.exceptions import StudyBuddyError
from src.llm import LocalLLM
from src.prompts import QA_PROMPT_TEMPLATE, STUDY_TUTOR_SYSTEM_PROMPT
from src.retriever import LocalRetriever
from src.schemas import AnswerResponse, SourceReference


class StudyChat:
    """Manages conversational question-answering grounded in student notes."""

    def __init__(
        self,
        retriever: Optional[LocalRetriever] = None,
        llm: Optional[LocalLLM] = None,
    ):
        self.retriever = retriever or LocalRetriever()
        self.llm = llm or LocalLLM()

    def format_context(self, sources: List[SourceReference]) -> str:
        """Format retrieved source chunks into untrusted reference text for the prompt."""
        if not sources:
            return "No matching study excerpts found."

        context_parts = []
        for idx, src in enumerate(sources, 1):
            page_info = f", Page {src.page_number}" if src.page_number else ""
            header = f"[Excerpt {idx}: {src.document_name}{page_info}]"
            context_parts.append(f"{header}\n{src.excerpt.strip()}")

        return "\n\n".join(context_parts)

    def answer_question(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> AnswerResponse:
        """Answer a student question using retrieved study passages.

        Args:
            question: Student's study question.
            doc_ids: Optional filter to query only specific indexed documents.
            top_k: Number of relevant chunks to retrieve.

        Returns:
            AnswerResponse schema containing text and verified citations.
        """
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve relevant passages locally
        sources = self.retriever.retrieve(
            query=clean_question,
            top_k=top_k,
            doc_ids=doc_ids,
        )

        # 2. Honest handling when no relevant context is found
        if not sources:
            return AnswerResponse(
                answer=(
                    "The uploaded study notes do not contain sufficient information "
                    "to answer this question. Try selecting more documents, rephrasing, "
                    "or uploading additional study materials."
                ),
                sources=[],
                has_sufficient_context=False,
            )

        # 3. Assemble context and prompt
        context_str = self.format_context(sources)
        prompt = QA_PROMPT_TEMPLATE.format(
            context=context_str,
            question=clean_question,
        )

        # 4. Generate answer using local LLM
        generated_answer = self.llm.generate(
            prompt=prompt,
            system_prompt=STUDY_TUTOR_SYSTEM_PROMPT,
        )

        return AnswerResponse(
            answer=generated_answer,
            sources=sources,
            has_sufficient_context=True,
        )

    def answer_question_stream(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> tuple[Generator[str, None, None], List[SourceReference], bool]:
        """Stream an answer response for reactive UI rendering.

        Returns:
            Tuple of (token_generator, sources_list, has_sufficient_context)
        """
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        sources = self.retriever.retrieve(
            query=clean_question,
            top_k=top_k,
            doc_ids=doc_ids,
        )

        if not sources:
            def empty_gen():
                yield (
                    "The uploaded study notes do not contain sufficient information "
                    "to answer this question. Try selecting more documents or uploading "
                    "additional notes."
                )

            return empty_gen(), [], False

        context_str = self.format_context(sources)
        prompt = QA_PROMPT_TEMPLATE.format(
            context=context_str,
            question=clean_question,
        )

        stream = self.llm.generate_stream(
            prompt=prompt,
            system_prompt=STUDY_TUTOR_SYSTEM_PROMPT,
        )
        return stream, sources, True
