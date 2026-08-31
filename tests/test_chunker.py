"""
Unit tests for document chunker module.
"""

import pytest

from app.config import settings
from app.ingestion.chunker import DocumentChunker
from langchain_core.documents import Document


def test_chunker_initialization():
    """Test chunker initialization."""
    chunker = DocumentChunker()
    assert chunker.chunk_size == settings.CHUNK_SIZE
    assert chunker.chunk_overlap == settings.CHUNK_OVERLAP


def test_chunker_custom_params():
    """Test chunker with custom parameters."""
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=50)
    assert chunker.chunk_size == 400
    assert chunker.chunk_overlap == 50


def test_chunk_documents():
    """Test document chunking."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    # Create a long document
    long_text = " ".join(["word"] * 200)
    doc = Document(page_content=long_text, metadata={"source": "test.pdf", "page": 1})
    
    chunks = chunker.chunk_documents([doc])
    
    assert len(chunks) > 1
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == "test.pdf" for chunk in chunks)


def test_chunk_metadata_preservation():
    """Test that chunk metadata is preserved."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    doc = Document(
        page_content="Test content " * 20,
        metadata={"source": "test.pdf", "page": 1, "document_id": "test"}
    )
    
    chunks = chunker.chunk_documents([doc])
    
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["document_id"] == "test"
        assert "chunk_id" in chunk.metadata
