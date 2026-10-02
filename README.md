# StudyBuddy Local 🎓

> **Tagline:** *Your private AI study partner. Your notes stay yours.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Local-First](https://img.shields.io/badge/Privacy-100%25%20Local-success.svg)](#11-privacy-design)
[![Hacktoberfest 2026](https://img.shields.io/badge/Hacktoberfest-Build%20for%20a%20Friend-orange.svg)](#19-hacktoberfest-2026-challenge-reference)
[![Tests Passing](https://img.shields.io/badge/pytest-42%20passed-brightgreen.svg)](#15-testing-instructions)

---

## 1. Project Overview & Tagline

**StudyBuddy Local** is a local-first, privacy-respecting AI study assistant designed to help students master complex subjects, revise for examinations, and practice grounded quizzes from their own course materials—**without uploading a single byte of personal documents to external cloud AI services**.

---

## 2. Problem Statement

Students routinely study using lengthy PDF lecture slides, textbooks, and personal class notes. Finding answers, cross-referencing concepts, and creating realistic practice test questions takes considerable manual effort.

While commercial cloud AI assistants exist, uploading unpublished lecture notes, university problem sets, or personal study documents to cloud servers introduces serious **privacy risks**, **recurring subscription/API fees**, and an **uncompromising dependence on constant internet connectivity**.

StudyBuddy Local resolves these problems by providing an open-weight, locally executed AI study copilot that performs on-device document extraction, local vector embeddings, grounded semantic retrieval, and local language model generation.

---

## 3. Intended User

Built for **Alex**, a close friend preparing for rigorous university examinations. Alex has hundreds of pages of PDF lecture slides, course notes, and homework problems. Alex wants an interactive study buddy to test comprehension, explain hard topics in simple terms, and create quizzes—all on an everyday laptop without risking academic data leakage or paying monthly API subscriptions.

---

## 4. Key Features

- **📄 Local Document Ingestion:** Safe extraction of text and page-level metadata from multi-page PDFs (via PyMuPDF) and plain text (`.txt`) files.
- **🔍 Sentence-Aware Chunking:** Smart segmentation respecting paragraph and sentence boundaries with configurable window sizes (800 characters) and overlap (120 characters).
- **⚡ Local Vector Storage (ChromaDB):** Embeddings generated on-device with Sentence Transformers and persisted locally in SQLite/ChromaDB.
- **💬 Grounded Study Chat:** Ask questions and receive clear explanations strictly grounded in uploaded notes, with expandable verified source citations (document name, page number, and similarity score).
- **🛡️ Prompt Injection Defense:** Treats all study materials as untrusted reference data, explicitly instructing the model never to execute commands embedded inside uploaded documents.
- **📝 Practice Quiz & Self-Testing:** Generates validated 4-option multiple-choice quizzes with difficulty controls, immediate correctness feedback, grounded explanations, and overall scorecard tracking.
- **📚 Multi-Tier Summaries & Cheat Sheets:**
  - *Short Summary:* Rapid 3-5 bullet point takeaways.
  - *Detailed Summary:* Deep-dive conceptual overview, mechanisms, and real-world examples.
  - *Revision Notes:* Exam cheat sheet featuring definitions, key concept comparisons, formulas, and self-test questions.
- **📂 Document Library Management:** View, inspect, deduplicate, and safely delete indexed documents with immediate vector index cleanup.
- **🔒 Auditable Local Privacy:** Strict localhost binding (`127.0.0.1`), zero external telemetry, zero cloud fallbacks.

---

## 5. User Interface & Demo

The application runs a clean, responsive, dark-themed Streamlit interface:

```text
+---------------------------------------------------------------------------------------+
|  StudyBuddy Local v1.0.0                                          [🔒 Local AI Mode] |
+-----------------------+---------------------------------------------------------------+
| 🟢 Ollama Connected   |  [💬 Study Chat]  [📝 Quiz]  [📚 Summaries]  [📂 Documents]    |
| Model: qwen2.5:3b     |---------------------------------------------------------------|
| Documents: 3 indexed  | Student: "What is the difference between paging & segmentation?"|
|                       |                                                               |
| Quick Upload:         | StudyBuddy Local:                                            |
| [Drag & drop PDF/TXT] | "Paging divides memory into fixed-size physical frames, while |
|                       |  segmentation divides memory into variable-sized logical     |
| [Manage Library]      |  units based on program modules..."                          |
|                       |                                                               |
|                       |  ▼ [📚 Verified Sources: os_lectures.pdf, Page 14 (94% match)]|
+-----------------------+---------------------------------------------------------------+
```

---

## 6. Technology Stack

- **User Interface:** [Streamlit](https://streamlit.io/) (bound strictly to `127.0.0.1` with usage statistics telemetry disabled)
- **Document Processing:** [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) and Python standard library
- **Local Embeddings:** [Sentence Transformers](https://sbert.net/) (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) (persistent local on-disk storage)
- **Local LLM Runtime:** [Ollama](https://ollama.com/) (local daemon over HTTP localhost)
- **Target Language Model:** Qwen2.5 3B Instruct (`qwen2.5:3b`)
- **Schema Validation:** [Pydantic v2](https://docs.pydantic.dev/)
- **Test Framework:** [pytest](https://docs.pytest.org/)

---

## 7. System Architecture

```text
                              STUDENT (Browser)
                                     |
                                     v
                       Streamlit Interface (127.0.0.1:8501)
                                     |
    +--------------------------------+--------------------------------+
    |                                |                                |
    v                                v                                v
[Document Loader]              [Study Chat RAG]              [Quiz & Summarizer]
    |                                |                                |
    v                                v                                v
[Text Chunker]               [Local Retriever]               [Pydantic Validation]
    |                                |                                |
    v                                v                                v
[Sentence Transformers] <------------+------------> [Ollama Local Runtime (11434)]
    |                                                                 |
    v                                                                 v
[ChromaDB Local Store]                                       [Qwen2.5 3B Open Weights]
 (data/chroma/)                                                (Zero Cloud Endpoints)
```

---

## 8. Installation Instructions

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- [Ollama](https://ollama.com/) installed on your machine.

### Windows (PowerShell)
```powershell
# 1. Clone the repository
git clone https://github.com/your-username/studybuddy-local.git
cd studybuddy-local

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local open-weight language model in Ollama
ollama pull qwen2.5:3b

# 5. Run the environment diagnostic script
python scripts/check_environment.py
```

### Linux / macOS (Bash)
```bash
# 1. Clone the repository
git clone https://github.com/your-username/studybuddy-local.git
cd studybuddy-local

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the model in Ollama
ollama pull qwen2.5:3b

# 5. Run diagnostics
python scripts/check_environment.py
```

---

## 9. Usage Instructions

1. **Start Ollama daemon** (if not already running as a system service):
   ```bash
   ollama serve
   ```

2. **Launch the StudyBuddy Local UI:**
   ```bash
   streamlit run app.py
   ```
   The browser will open automatically at `http://127.0.0.1:8501`.

3. **Ingest Notes:** Navigate to **📂 My Documents** or use the sidebar uploader to upload your lecture notes or textbooks in PDF or TXT format.
4. **Chat & Study:** Switch to **💬 Study Chat** and ask conceptual questions. Expand the source excerpts to cross-check cited page numbers.
5. **Take Practice Quizzes:** Switch to **📝 Quiz & Practice**, configure difficulty, and test your knowledge with instant answers and scorecards.

---

## 10. Local Model Configuration

Configuration values are stored in [`config.py`](file:///C:/Users/Utkarsh/OneDrive/Desktop/HACKTOBERFEST%20TASK%201/config.py) and can be overridden via environment variables without modifying source code:

| Setting | Default Value | Environment Variable |
| :--- | :--- | :--- |
| **Ollama URL** | `http://127.0.0.1:11434` | `OLLAMA_BASE_URL` |
| **LLM Model** | `qwen2.5:3b` | `OLLAMA_MODEL` |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | `EMBEDDING_MODEL_NAME` |
| **Chunk Size** | `800` characters | `CHUNK_SIZE` |
| **Chunk Overlap** | `120` characters | `CHUNK_OVERLAP` |
| **Top-K Retrieval**| `5` passages | `TOP_K_CHUNKS` |
| **Relevance Threshold**| `0.20` cosine similarity | `RELEVANCE_THRESHOLD` |
| **Max File Upload**| `25` MB | `MAX_UPLOAD_SIZE_MB` |

Compatible alternative local models: `llama3.2:3b`, `phi3:mini`, `mistral:7b`.

---

## 11. Privacy Design

Privacy is a non-negotiable core design requirement in StudyBuddy Local:

- **Strict Localhost Enforcement:** [`src/privacy.py`](file:///C:/Users/Utkarsh/OneDrive/Desktop/HACKTOBERFEST%20TASK%201/src/privacy.py) asserts that all service endpoints resolve strictly to loopback addresses (`127.0.0.1`). Any attempt to configure external cloud AI domains (OpenAI, Google Cloud, Anthropic, Cohere, Pinecone) triggers an immediate `PrivacyViolationError`.
- **Zero Outbound Telemetry:** Streamlit usage statistics and third-party tracking are disabled in `.streamlit/config.toml` (`gatherUsageStats = false`).
- **No Cloud Fallbacks:** If the local model daemon is offline, the application reports a clear status message instead of secretly routing data to a cloud API.
- **Log Sanitization:** Sensitive note contents and student prompts are sanitized and never written unmasked to system logs.
- **Excluded from Version Control:** All vector databases (`data/chroma`) and uploaded notes are blocked in `.gitignore`.

---

## 12. Offline Verification Procedure

To verify that StudyBuddy Local operates completely independently of an internet connection:

1. Complete the installation and download `qwen2.5:3b` via Ollama.
2. Ingest a sample PDF or TXT study document while connected.
3. **Completely disconnect your computer from the internet** (turn off Wi-Fi and unplug Ethernet).
4. Launch the application: `streamlit run app.py`.
5. Ask questions in **Study Chat** and generate a practice **Quiz**.
6. Verify that retrieval, answer synthesis, and quiz generation function seamlessly with zero internet access.

---

## 13. Hardware Limitations

StudyBuddy Local was specifically engineered to run efficiently on an everyday laptop:
- **RAM Target:** Runs comfortably within 8 GB to 16 GB of system RAM.
- **Model Footprint:** The default `qwen2.5:3b` model consumes approximately ~2.0 GB of memory.
- **Embedding Footprint:** The `all-MiniLM-L6-v2` embedding model requires only ~120 MB of RAM.
- **Context Windows:** Bounded context sizes (top-5 chunks) prevent out-of-memory errors on CPU-only machines.

---

## 14. Known Limitations

- **Scanned PDFs / OCR:** The MVP extracts embedded text streams. Image-only or scanned PDFs without embedded text are flagged with an actionable warning; optical character recognition (OCR) is not included in this offline version.
- **Model Hallucination:** While strict RAG prompt constraints instruct the model never to invent citations, smaller local models (3B) can occasionally misinterpret complex tables or diagrams.
- **Single-User Local Session:** Session memory is maintained in the current browser session state; persistent cross-session chat databases are omitted in the MVP.

---

## 15. Testing Instructions

StudyBuddy Local features an automated unit test suite runnable without downloading model weights or running external services:

```bash
# Run the complete test suite
pytest tests/ -v
```

Current test coverage includes **42 automated tests**:
- `tests/test_document_loader.py`: Valid PDF/TXT, empty files, scanned PDFs, corrupted bytes, oversized files.
- `tests/test_chunker.py`: Boundary splitting, overlap correctness, metadata preservation.
- `tests/test_llm.py`: Connection timeout handling, missing model recovery, JSON extraction.
- `tests/test_privacy.py`: Localhost assertions, blocked cloud patterns, log sanitization.
- `tests/test_retriever.py`: Vector search, document filtering by ID, document deletion.
- `tests/test_study_chat.py`: Grounded QA, insufficient context handling, token streaming.
- `tests/test_quiz_generator.py`: Schema validation, single-retry recovery, scoring calculations.

---

## 16. Model Licensing Information

- **Qwen2.5 3B Instruct (`qwen2.5:3b`):** Released under the permissive **Apache 2.0 License** by the Qwen Team / Alibaba Cloud. Permitted for both personal and commercial open-source use.
- **all-MiniLM-L6-v2:** Released under the **Apache 2.0 License** by Sentence Transformers / Hugging Face.

---

## 17. Third-Party Dependency Licenses

- `streamlit`: Apache 2.0 License
- `PyMuPDF (fitz)`: GNU AGPL v3.0 / Commercial License
- `chromadb`: Apache 2.0 License
- `sentence-transformers`: Apache 2.0 License
- `pydantic`: MIT License
- `ollama-python`: MIT License
- `pytest`: MIT License

---

## 18. Contribution Instructions

Contributions are welcome! Please follow these guidelines:
1. Fork the repository and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Ensure new features are strictly local-first (never add cloud AI fallbacks).
3. Add unit tests for any new modules in `tests/`.
4. Run `pytest tests/ -v` to ensure all tests pass.
5. Submit a descriptive Pull Request.

---

## 19. Hacktoberfest 2026 Challenge Reference

- **Event:** Hacktoberfest 2026 Weekend Challenge
- **Theme:** *Build for a Friend*
- **Built For:** A real friend preparing for university examinations.
- **Why Open Innovation Matters for this Project:**
  - **Uncompromising Privacy:** Academic documents, unreleased homework, and thesis notes remain exclusively on the student's hardware.
  - **Zero Subscription Barrier:** Eliminates monthly cloud AI subscription fees ($20/month) for students on a budget.
  - **Works Anywhere Offline:** Allows studying in lecture halls, libraries, and transit with unreliable internet connectivity.
  - **Open Auditing:** Complete transparency in how study passages are chunked, embedded, and retrieved.

---

## 20. Project Code License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.
