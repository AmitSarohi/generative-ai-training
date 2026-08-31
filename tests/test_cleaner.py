"""
Unit tests for text cleaner module.
"""

import pytest

from app.ingestion.cleaner import TextCleaner
from langchain_core.documents import Document


def test_cleaner_initialization():
    """Test text cleaner initialization."""
    cleaner = TextCleaner()
    assert cleaner is not None


def test_clean_text():
    """Test text cleaning."""
    cleaner = TextCleaner()
    
    # Test excessive whitespace
    text = "This  has    excessive   whitespace"
    cleaned = cleaner.clean_text(text)
    assert "  " not in cleaned
    assert cleaned == "This has excessive whitespace"
    
    # Test leading/trailing whitespace
    text = "  text with spaces  "
    cleaned = cleaner.clean_text(text)
    assert cleaned == "text with spaces"


def test_clean_document():
    """Test document cleaning."""
    cleaner = TextCleaner()
    doc = Document(page_content="  Test content  ", metadata={"page": 1})
    
    cleaned_doc = cleaner.clean_document(doc)
    assert cleaned_doc.page_content == "Test content"
    assert cleaned_doc.metadata == {"page": 1}


def test_clean_documents():
    """Test cleaning multiple documents."""
    cleaner = TextCleaner()
    docs = [
        Document(page_content="  Doc 1  ", metadata={"page": 1}),
        Document(page_content="  Doc 2  ", metadata={"page": 2})
    ]
    
    cleaned_docs = cleaner.clean_documents(docs)
    assert len(cleaned_docs) == 2
    assert cleaned_docs[0].page_content == "Doc 1"
    assert cleaned_docs[1].page_content == "Doc 2"
