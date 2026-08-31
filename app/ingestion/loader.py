"""
PDF loader module using LangChain's PyMuPDFLoader.
Loads the five required PDF documents from data/pdfs/.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from app.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads PDF documents using PyMuPDFLoader."""
    
    def __init__(self, pdfs_dir: Path = None):
        """
        Initialize PDF loader.
        
        Args:
            pdfs_dir: Directory containing PDF files. Defaults to settings.PDFS_DIR.
        """
        self.pdfs_dir = pdfs_dir or settings.PDFS_DIR
        self.required_pdfs = settings.REQUIRED_PDFS
    
    def load_pdf(self, pdf_filename: str) -> List[Document]:
        """
        Load a single PDF file.
        
        Args:
            pdf_filename: Name of the PDF file to load.
            
        Returns:
            List of LangChain Document objects.
            
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            Exception: If PDF loading fails.
        """
        pdf_path = self.pdfs_dir / pdf_filename
        
        if not pdf_path.exists():
            error_msg = f"Required PDF {pdf_filename} was not found in {self.pdfs_dir}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            logger.info(f"Loading PDF: {pdf_filename}")
            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # Add metadata to each document
            document_id = pdf_filename.replace(".pdf", "")
            for doc in documents:
                doc.metadata["source"] = pdf_filename
                doc.metadata["filename"] = pdf_filename
                doc.metadata["document_id"] = document_id
            
            logger.info(f"Loaded {len(documents)} pages from {pdf_filename}")
            return documents
            
        except Exception as e:
            error_msg = f"Failed to load PDF {pdf_filename}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def load_all_pdfs(self) -> List[Document]:
        """
        Load all five required PDF documents.
        
        Returns:
            List of all LangChain Document objects from all PDFs.
            
        Raises:
            Exception: If any required PDF is missing or fails to load.
        """
        all_documents = []
        
        for pdf_filename in self.required_pdfs:
            try:
                documents = self.load_pdf(pdf_filename)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading {pdf_filename}: {str(e)}")
                raise
        
        logger.info(f"Total documents loaded: {len(all_documents)} pages")
        return all_documents
    
    def verify_pdfs_exist(self) -> bool:
        """
        Verify that all required PDFs exist in the PDFs directory.
        
        Returns:
            True if all PDFs exist, False otherwise.
        """
        missing_pdfs = []
        for pdf_filename in self.required_pdfs:
            pdf_path = self.pdfs_dir / pdf_filename
            if not pdf_path.exists():
                missing_pdfs.append(pdf_filename)
        
        if missing_pdfs:
            logger.error(f"Missing PDFs: {', '.join(missing_pdfs)}")
            return False
        
        logger.info("All required PDFs are present.")
        return True
