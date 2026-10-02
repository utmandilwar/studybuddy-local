"""Chapter summary and exam revision notes generation for StudyBuddy Local.

Generates bounded, grounded summaries across 3 distinct student modes:
1. Short summary (quick overview)
2. Detailed summary (comprehensive deep-dive)
3. Revision notes (exam formulas, key comparisons, and high-yield cheat sheets)
"""

from __future__ import annotations

from typing import List, Optional

from config import AppConfig
from src.llm import LocalLLM
from src.prompts import (
    DETAILED_SUMMARY_PROMPT_TEMPLATE,
    REVISION_NOTES_PROMPT_TEMPLATE,
    SHORT_SUMMARY_PROMPT_TEMPLATE,
    STUDY_TUTOR_SYSTEM_PROMPT,
)
from src.retriever import LocalRetriever
from src.schemas import AnswerResponse, SourceReference
from src.vector_store import LocalVectorStore


class SummaryGenerator:
    """Produces multi-tier summaries and exam revision notes from local study material."""

    def __init__(
        self,
        retriever: Optional[LocalRetriever] = None,
        llm: Optional[LocalLLM] = None,
        vector_store: Optional[LocalVectorStore] = None,
    ):
        self.retriever = retriever or LocalRetriever()
        self.llm = llm or LocalLLM()
        self.vector_store = vector_store or LocalVectorStore()

    def _collect_representative_chunks(
        self,
        doc_ids: Optional[List[str]] = None,
        topic_focus: Optional[str] = None,
        max_chunks: int = 6,
    ) -> List[SourceReference]:
        """Gather representative chunks for summarization without exceeding context limits."""
        if topic_focus and topic_focus.strip():
            # Focused semantic retrieval
            return self.retriever.retrieve(
                query=topic_focus.strip(),
                top_k=max_chunks,
                doc_ids=doc_ids,
            )

        # Broad general retrieval: query for central concepts
        general_queries = ["key concepts definitions overview principles summary"]
        for q in general_queries:
            chunks = self.retriever.retrieve(
                query=q,
                top_k=max_chunks,
                doc_ids=doc_ids,
                relevance_threshold=0.05,  # wider threshold for general collection
            )
            if chunks:
                return chunks

        return []

    def generate_summary(
        self,
        doc_ids: Optional[List[str]] = None,
        mode: str = "short",
        topic_focus: Optional[str] = None,
    ) -> AnswerResponse:
        """Generate a grounded study summary or revision cheat sheet.

        Args:
            doc_ids: Optional document filters.
            mode: 'short', 'detailed', or 'revision'.
            topic_focus: Optional sub-topic to focus the summary on.

        Returns:
            AnswerResponse containing generated summary and cited sources.
        """
        valid_modes = {"short", "detailed", "revision"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid summary mode '{mode}'. Choose from: {valid_modes}")

        sources = self._collect_representative_chunks(
            doc_ids=doc_ids,
            topic_focus=topic_focus,
            max_chunks=6,
        )

        if not sources:
            return AnswerResponse(
                answer=(
                    "No indexed study material available to summarize. "
                    "Please upload a document first in 'My Documents'."
                ),
                sources=[],
                has_sufficient_context=False,
            )

        # Format context excerpts
        context_parts = []
        for idx, src in enumerate(sources, 1):
            page_info = f", Page {src.page_number}" if src.page_number else ""
            context_parts.append(f"[Source {idx}: {src.document_name}{page_info}]\n{src.excerpt}")
        context_str = "\n\n".join(context_parts)

        # Select corresponding prompt template
        if mode == "short":
            prompt = SHORT_SUMMARY_PROMPT_TEMPLATE.format(context=context_str)
        elif mode == "detailed":
            prompt = DETAILED_SUMMARY_PROMPT_TEMPLATE.format(context=context_str)
        else:  # revision
            prompt = REVISION_NOTES_PROMPT_TEMPLATE.format(context=context_str)

        generated_text = self.llm.generate(
            prompt=prompt,
            system_prompt=STUDY_TUTOR_SYSTEM_PROMPT,
        )

        return AnswerResponse(
            answer=generated_text,
            sources=sources,
            has_sufficient_context=True,
        )
