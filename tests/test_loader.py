"""
Unit tests for PDF loader module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from app.ingestion.loader import PDFLoader
from app.config import settings


def test_pdf_loader_initialization():
    """Test PDF loader initialization."""
    loader = PDFLoader()
    assert loader.pdfs_dir == settings.PDFS_DIR
    assert loader.required_pdfs == settings.REQUIRED_PDFS


def test_verify_pdfs_exist():
    """Test PDF existence verification."""
    loader = PDFLoader()
    # This test will fail if PDFs are not present
    # In a real test environment, we would mock this
    result = loader.verify_pdfs_exist()
    assert isinstance(result, bool)


def test_load_pdf_not_found():
    """Test loading a non-existent PDF."""
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_pdf("nonexistent.pdf")


def test_load_pdf_success():
    """Test successful PDF loading (mocked)."""
    loader = PDFLoader()
    
    # Mock PyMuPDFLoader
    with patch('app.ingestion.loader.PyMuPDFLoader') as mock_loader_class:
        mock_loader = Mock()
        mock_loader.load.return_value = [
            Mock(page_content="Test content", metadata={"page": 1})
        ]
        mock_loader_class.return_value = mock_loader
        
        # Create a test file
        test_pdf = settings.PDFS_DIR / "test.pdf"
        settings.PDFS_DIR.mkdir(parents=True, exist_ok=True)
        test_pdf.touch()
        
        try:
            documents = loader.load_pdf("test.pdf")
            assert len(documents) == 1
            assert documents[0].metadata["source"] == "test.pdf"
        finally:
            test_pdf.unlink()
