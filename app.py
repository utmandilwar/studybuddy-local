"""StudyBuddy Local — Streamlit User Interface.

Your private AI study partner. Your notes stay yours.
Strictly local-first: zero cloud inference, zero external telemetry.
"""

from __future__ import annotations

import streamlit as st

from config import AppConfig
from src.document_manager import DocumentManager
from src.llm import LocalLLM
from src.privacy import (
    assert_local_configuration,
    get_privacy_badge,
    get_privacy_disclosure,
)
from src.quiz_generator import QuizGenerator, QuizSession
from src.study_chat import StudyChat
from src.summary_generator import SummaryGenerator

# Page configuration
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} — Private AI Study Partner",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize application components
@st.cache_resource
def get_services():
    """Initialize shared backend service singletons."""
    doc_mgr = DocumentManager()
    llm = LocalLLM()
    chat = StudyChat(retriever=None, llm=llm)
    quiz_gen = QuizGenerator(retriever=None, llm=llm)
    summarizer = SummaryGenerator(retriever=None, llm=llm)
    return doc_mgr, llm, chat, quiz_gen, summarizer


doc_manager, llm_client, study_chat, quiz_generator, summary_generator = get_services()

# Session State Initialization
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "active_quiz_session" not in st.session_state:
    st.session_state.active_quiz_session = None

if "quiz_answers_state" not in st.session_state:
    st.session_state.quiz_answers_state = {}


def render_sidebar():
    """Render the sidebar with connection status, uploader, and privacy badge."""
    with st.sidebar:
        st.markdown(f"## 🎓 {AppConfig.APP_NAME}")
        st.caption(AppConfig.TAGLINE)
        st.divider()

        # Privacy Badge
        badge_text, is_verified = get_privacy_badge()
        if is_verified:
            st.success(f"🔒 **{badge_text}**")
        else:
            st.error(f"⚠️ {badge_text}")

        st.divider()

        # Ollama Health & Model Status
        st.markdown("### 🖥️ Local AI Status")
        is_connected, status_msg, available_models = llm_client.check_connection()

        if is_connected:
            st.success(f"🟢 **Ollama Connected**\n\nModel: `{AppConfig.DEFAULT_MODEL}`")
            if available_models:
                with st.expander("Downloaded Local Models"):
                    for m in available_models:
                        st.markdown(f"- `{m}`")
        else:
            st.warning("🟡 **Ollama Offline**")
            st.caption(status_msg)
            with st.expander("How to start Ollama"):
                st.code(
                    "1. Install: https://ollama.com\n"
                    "2. Run: ollama serve\n"
                    f"3. Model: ollama pull {AppConfig.DEFAULT_MODEL}",
                    language="bash",
                )

        st.divider()

        # Quick Document Uploader
        st.markdown("### 📄 Quick Upload Notes")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help=f"Max {AppConfig.MAX_UPLOAD_SIZE_MB}MB per file. Extracted locally.",
        )

        if uploaded_files:
            for file in uploaded_files:
                file_bytes = file.read()
                # Check if already indexed
                existing_docs = {d.name: d for d in doc_manager.list_documents()}
                if file.name not in existing_docs:
                    with st.spinner(f"Indexing '{file.name}' locally..."):
                        try:
                            meta, count = doc_manager.process_and_index_document(
                                file_bytes, file.name
                            )
                            st.success(f"Indexed **{file.name}** ({count} passages)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error indexing {file.name}: {e}")

        # Document Stats
        docs = doc_manager.list_documents()
        st.metric("Indexed Documents", len(docs))


def render_study_chat():
    """Render the primary Study Chat interface."""
    st.markdown("### 💬 Study Chat")
    st.caption("Ask questions about your uploaded lecture notes, textbooks, and study sheets.")

    docs = doc_manager.list_documents()
    if not docs:
        st.info(
            "👉 **No study notes uploaded yet!** "
            "Please upload your study PDF or TXT file using the sidebar or the **My Documents** tab to begin."
        )

    # Document selection filter
    doc_options = {d.name: d.doc_id for d in docs}
    selected_doc_names = st.multiselect(
        "Filter by document (optional):",
        options=list(doc_options.keys()),
        default=[],
        placeholder="Searching all indexed documents...",
    )
    selected_doc_ids = [doc_options[name] for name in selected_doc_names] if selected_doc_names else None

    # Chat history display
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Verified Source Passages", expanded=False):
                    for idx, src in enumerate(msg["sources"], 1):
                        page_str = f", Page {src['page_number']}" if src.get("page_number") else ""
                        st.markdown(
                            f"**[{idx}] {src['document_name']}{page_str}** "
                            f"(Relevance: {int(src['score'] * 100)}%)"
                        )
                        st.caption(f"> \"{src['excerpt']}\"")

    # Chat input
    user_query = st.chat_input("Ask a question about your study material...")
    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching local notes & reasoning..."):
                try:
                    resp = study_chat.answer_question(
                        question=user_query,
                        doc_ids=selected_doc_ids,
                    )
                    st.markdown(resp.answer)

                    # Serialize sources for session state
                    sources_data = [s.model_dump() for s in resp.sources]
                    if resp.sources:
                        with st.expander("📚 Verified Source Passages", expanded=True):
                            for idx, src in enumerate(resp.sources, 1):
                                page_str = f", Page {src.page_number}" if src.page_number else ""
                                st.markdown(
                                    f"**[{idx}] {src.document_name}{page_str}** "
                                    f"(Relevance: {int((src.score or 0) * 100)}%)"
                                )
                                st.caption(f"> \"{src.excerpt}\"")

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": resp.answer,
                            "sources": sources_data,
                        }
                    )
                except Exception as err:
                    st.error(f"Error generating answer: {err}")

    # Action bar
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()


def render_quiz_practice():
    """Render the practice quiz interface."""
    st.markdown("### 📝 Practice Quiz & Self-Testing")
    st.caption("Generate grounded multiple-choice questions from your notes to test exam readiness.")

    docs = doc_manager.list_documents()
    if not docs:
        st.warning("Please upload study notes before generating a quiz.")
        return

    doc_options = {d.name: d.doc_id for d in docs}

    with st.expander("⚙️ Quiz Configuration", expanded=(st.session_state.active_quiz_session is None)):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_docs = st.multiselect(
                "Source documents:",
                options=list(doc_options.keys()),
                default=list(doc_options.keys())[:1],
            )
        with col2:
            num_q = st.slider("Number of questions:", min_value=1, max_value=10, value=3)
        with col3:
            difficulty = st.selectbox("Difficulty:", ["Easy", "Medium", "Hard"], index=1)

        topic_focus = st.text_input(
            "Specific topic or chapter (optional):",
            placeholder="e.g. Memory Management, CPU Scheduling...",
        )

        selected_ids = [doc_options[n] for n in selected_docs] if selected_docs else None

        if st.button("🚀 Generate Practice Quiz", type="primary", use_container_width=True):
            with st.spinner("Generating grounded multiple-choice questions locally..."):
                try:
                    generated_quiz = quiz_generator.generate_quiz(
                        doc_ids=selected_ids,
                        num_questions=num_q,
                        difficulty=difficulty,
                        topic_focus=topic_focus,
                    )
                    st.session_state.active_quiz_session = QuizSession(generated_quiz)
                    st.session_state.quiz_answers_state = {}
                    st.rerun()
                except Exception as err:
                    st.error(f"Quiz generation failed: {err}")

    session: QuizSession = st.session_state.active_quiz_session
    if session is not None:
        st.divider()
        st.markdown(f"#### Quiz in Progress ({session.total_questions} Questions)")

        for idx, q in enumerate(session.quiz.questions):
            st.markdown(f"**Question {idx + 1}:** {q.question}")

            options = [f"{k}: {v}" for k, v in q.options.items()]
            current_choice = st.session_state.quiz_answers_state.get(idx)

            user_selection = st.radio(
                f"Choose answer for Question {idx + 1}:",
                options=options,
                index=None if current_choice is None else options.index(current_choice),
                key=f"q_radio_{idx}",
                disabled=session.is_submitted(idx),
            )

            if not session.is_submitted(idx):
                if st.button(f"Submit Answer for Q{idx + 1}", key=f"btn_sub_{idx}"):
                    if user_selection is None:
                        st.warning("Please pick an option first.")
                    else:
                        selected_key = user_selection.split(":")[0].strip()
                        session.submit_answer(idx, selected_key)
                        st.session_state.quiz_answers_state[idx] = user_selection
                        st.rerun()
            else:
                user_choice = session.get_user_answer(idx)
                if user_choice == q.correct_answer:
                    st.success(f"✅ **Correct!** ({q.correct_answer})")
                else:
                    st.error(f"❌ **Incorrect.** You selected `{user_choice}`, correct answer is `{q.correct_answer}`.")

                st.info(f"💡 **Explanation:** {q.explanation}")
                page_str = f", Page {q.source_page}" if q.source_page else ""
                st.caption(f"Source: `{q.source_document}{page_str}`")

            st.write("")

        # Scorecard
        if session.is_complete():
            st.divider()
            score_data = session.calculate_score()
            st.balloons()
            st.markdown("### 🏆 Quiz Completed!")
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Score", f"{score_data['correct']} / {score_data['total']}")
            col_s2.metric("Percentage", f"{score_data['percentage']}%")
            col_s3.metric("Status", "Passed" if score_data['percentage'] >= 70 else "Needs Revision")

            if st.button("🔄 Restart Quiz", use_container_width=True):
                session.reset()
                st.session_state.quiz_answers_state = {}
                st.rerun()


def render_summaries_revision():
    """Render chapter summaries and exam revision notes."""
    st.markdown("### 📚 Chapter Summaries & Revision Cheat Sheets")
    st.caption("Generate high-yield summaries and revision notes grounded in your study material.")

    docs = doc_manager.list_documents()
    if not docs:
        st.warning("Please upload study notes before generating summaries.")
        return

    doc_options = {d.name: d.doc_id for d in docs}

    col1, col2 = st.columns(2)
    with col1:
        selected_docs = st.multiselect(
            "Select study documents:",
            options=list(doc_options.keys()),
            default=list(doc_options.keys())[:1],
        )
    with col2:
        mode = st.selectbox(
            "Summary type:",
            options=["short", "detailed", "revision"],
            format_func=lambda m: {
                "short": "⚡ Short Summary (Quick 3-5 Key Bullets)",
                "detailed": "📖 Detailed Study Summary (Comprehensive Overview)",
                "revision": "📌 Exam Revision Cheat Sheet (Formulas, Comparisons & Tips)",
            }[m],
        )

    topic_focus = st.text_input(
        "Focus on specific chapter or sub-topic (optional):",
        placeholder="e.g. Chapter 3, Process Synchronization...",
    )

    selected_ids = [doc_options[n] for n in selected_docs] if selected_docs else None

    if st.button("✨ Generate Summary", type="primary", use_container_width=True):
        with st.spinner("Synthesizing grounded summary with local model..."):
            try:
                resp = summary_generator.generate_summary(
                    doc_ids=selected_ids,
                    mode=mode,
                    topic_focus=topic_focus,
                )
                st.markdown("---")
                st.markdown(resp.answer)

                if resp.sources:
                    with st.expander("📚 Grounded Source References", expanded=False):
                        for idx, src in enumerate(resp.sources, 1):
                            page_str = f", Page {src.page_number}" if src.page_number else ""
                            st.markdown(f"**[{idx}] {src.document_name}{page_str}**")
                            st.caption(f"> \"{src.excerpt}\"")
            except Exception as e:
                st.error(f"Summary generation error: {e}")


def render_my_documents():
    """Render the document library management view."""
    st.markdown("### 📂 My Document Library")
    st.caption("Inspect, manage, and delete your locally indexed notes.")

    # Ingestion Form
    with st.expander("➕ Upload New Study Document", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a PDF or plain text (.txt) file",
            type=["pdf", "txt"],
            key="doc_page_uploader",
        )
        if uploaded_file and st.button("Process & Index Document", type="primary"):
            file_bytes = uploaded_file.read()
            with st.spinner(f"Extracting & indexing '{uploaded_file.name}' locally..."):
                try:
                    meta, count = doc_manager.process_and_index_document(
                        file_bytes, uploaded_file.name
                    )
                    st.success(
                        f"Successfully indexed **{meta.name}**! "
                        f"({meta.page_count} pages, {count} searchable passages)"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process document: {e}")

    # Documents List
    docs = doc_manager.list_documents()
    if not docs:
        st.info("No documents indexed yet. Upload a PDF or TXT above to begin.")
        return

    st.markdown("#### Indexed Study Materials")
    for doc in docs:
        size_kb = round(doc.size_bytes / 1024, 1)
        with st.container():
            col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
            with col1:
                st.markdown(f"📄 **{doc.name}**")
                st.caption(f"Added: {doc.uploaded_at}")
            with col2:
                st.write(f"**{doc.page_count}** pages")
                st.caption(f"{size_kb} KB")
            with col3:
                st.write(f"**{doc.chunk_count}** passages")
                st.caption("Indexed in ChromaDB")
            with col4:
                if st.button("🗑️ Delete", key=f"del_{doc.doc_id}", type="secondary"):
                    doc_manager.delete_document(doc.doc_id)
                    st.success(f"Removed {doc.name}")
                    st.rerun()
            st.divider()


def render_privacy_transparency():
    """Render the privacy guarantees and technical transparency view."""
    st.markdown("### 🔒 Privacy Architecture & Offline Verification")
    st.caption("Auditable local-first guarantees: your study notes never leave this machine.")

    badge_text, is_verified = get_privacy_badge()
    if is_verified:
        st.success(f"### ✅ {badge_text}")
    else:
        st.error(f"### ⚠️ {badge_text}")

    disclosure = get_privacy_disclosure()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. Local-Only Execution")
        st.info(disclosure["local_execution"])

        st.markdown("#### 2. Local Embeddings & Vectors")
        st.info(disclosure["local_embeddings"])

    with col2:
        st.markdown("#### 3. Persistent Local Storage")
        st.info(disclosure["local_storage"])

        st.markdown("#### 4. Zero Telemetry")
        st.info(disclosure["zero_telemetry"])

    st.markdown("#### 5. Verification Boundary & Offline Acceptance Test")
    st.warning(disclosure["transparency_boundary"])

    st.markdown(
        """
        #### How to Verify Offline Operation Yourself:
        1. Ensure models are downloaded (`ollama pull qwen2.5:3b`).
        2. Upload your sample study documents into StudyBuddy Local.
        3. **Disconnect your laptop from Wi-Fi / Ethernet completely.**
        4. Ask questions in **Study Chat** and generate a practice **Quiz**.
        5. Notice that all embeddings, semantic searches, and answer generation continue working normally with zero network connection.
        """
    )


# Main Navigation Tabs
def main():
    render_sidebar()

    tabs = st.tabs(
        [
            "💬 Study Chat",
            "📝 Quiz & Practice",
            "📚 Summaries & Revision",
            "📂 My Documents",
            "🔒 Privacy & Architecture",
        ]
    )

    with tabs[0]:
        render_study_chat()

    with tabs[1]:
        render_quiz_practice()

    with tabs[2]:
        render_summaries_revision()

    with tabs[3]:
        render_my_documents()

    with tabs[4]:
        render_privacy_transparency()


if __name__ == "__main__":
    main()
