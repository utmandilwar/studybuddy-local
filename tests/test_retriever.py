"""Tests for LocalVectorStore, LocalRetriever, and DocumentManager."""

import pytest
from unittest.mock import MagicMock

from src.document_manager import DocumentManager
from src.retriever import LocalRetriever
from src.schemas import DocumentChunk
from src.vector_store import LocalVectorStore


@pytest.fixture
def temp_vector_store(tmp_path):
    """Create an isolated temporary ChromaDB vector store."""
    persist_dir = tmp_path / "test_chroma"
    return LocalVectorStore(persist_directory=persist_dir, collection_name="test_notes")


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager that returns fixed 4-dimensional normalized vectors."""
    manager = MagicMock()
    # Mock text embeddings: returns vector based on keyword presence
    def fake_embed(texts, batch_size=16):
        vectors = []
        for t in texts:
            if "kernel" in t.lower() or "operating" in t.lower():
                vectors.append([0.9, 0.1, 0.0, 0.0])
            elif "database" in t.lower() or "sql" in t.lower():
                vectors.append([0.0, 0.0, 0.9, 0.1])
            else:
                vectors.append([0.25, 0.25, 0.25, 0.25])
        return vectors

    manager.generate_embeddings.side_effect = fake_embed
    manager.generate_query_embedding.side_effect = lambda q: fake_embed([q])[0]
    return manager


def test_vector_store_add_and_query(temp_vector_store, mock_embedding_manager):
    """Test adding chunks and semantic retrieval in local vector store."""
    chunks = [
        DocumentChunk(
            chunk_id="doc1_p1_c0",
            doc_id="doc1",
            document_name="os.txt",
            page_number=1,
            chunk_index=0,
            text="The operating system kernel runs in privileged supervisor mode.",
        ),
        DocumentChunk(
            chunk_id="doc2_p1_c0",
            doc_id="doc2",
            document_name="db.txt",
            page_number=1,
            chunk_index=0,
            text="Relational database tables use SQL queries and B-tree indexes.",
        ),
    ]

    embs = mock_embedding_manager.generate_embeddings([c.text for c in chunks])
    indexed = temp_vector_store.add_chunks(chunks, embs)
    assert indexed == 2
    assert temp_vector_store.get_total_chunk_count() == 2

    # Query OS notes
    os_query_vec = mock_embedding_manager.generate_query_embedding("What is the kernel mode?")
    res = temp_vector_store.query(os_query_vec, top_k=1)
    retrieved_text = res["documents"][0][0]
    assert "operating system kernel" in retrieved_text


def test_retriever_document_filtering(temp_vector_store, mock_embedding_manager):
    """Test restricting retrieval to specific document IDs."""
    chunks = [
        DocumentChunk(
            chunk_id="doc1_c0",
            doc_id="doc1",
            document_name="doc1.txt",
            page_number=1,
            chunk_index=0,
            text="Operating system memory management algorithms.",
        ),
        DocumentChunk(
            chunk_id="doc2_c0",
            doc_id="doc2",
            document_name="doc2.txt",
            page_number=1,
            chunk_index=0,
            text="Operating system CPU scheduling algorithms.",
        ),
    ]
    embs = mock_embedding_manager.generate_embeddings([c.text for c in chunks])
    temp_vector_store.add_chunks(chunks, embs)

    retriever = LocalRetriever(
        vector_store=temp_vector_store, embedding_manager=mock_embedding_manager
    )

    # Filter to only doc2
    results = retriever.retrieve("Operating system", top_k=5, doc_ids=["doc2"])
    assert len(results) == 1
    assert results[0].document_name == "doc2.txt"


def test_document_manager_lifecycle(tmp_path, temp_vector_store, mock_embedding_manager):
    """Test full document lifecycle: process, deduplicate, list, delete."""
    meta_file = tmp_path / "documents.json"
    doc_manager = DocumentManager(
        metadata_file=meta_file,
        vector_store=temp_vector_store,
        embedding_manager=mock_embedding_manager,
    )

    sample_doc_content = b"Computer Architecture lecture 1: ALU and register files."
    meta, count = doc_manager.process_and_index_document(
        sample_doc_content, "arch_lecture.txt"
    )

    assert meta.name == "arch_lecture.txt"
    assert count >= 1
    assert len(doc_manager.list_documents()) == 1

    # Re-upload the exact same file (should update without duplicate records)
    meta2, count2 = doc_manager.process_and_index_document(
        sample_doc_content, "arch_lecture.txt"
    )
    assert meta2.doc_id == meta.doc_id
    assert len(doc_manager.list_documents()) == 1

    # Delete document
    deleted = doc_manager.delete_document(meta.doc_id)
    assert deleted is True
    assert len(doc_manager.list_documents()) == 0
    assert temp_vector_store.get_total_chunk_count() == 0
