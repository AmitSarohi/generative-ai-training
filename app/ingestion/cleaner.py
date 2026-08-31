"""
Text preprocessing module for cleaning extracted PDF text.
Performs lightweight preprocessing to prepare text for chunking.
"""

import logging
import re
from typing import List

from langchain_core.documents import Document

from app.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and preprocesses extracted text from PDFs."""
    
    def __init__(self):
        """Initialize text cleaner."""
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return text
        
        # Normalize excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_document(self, document: Document) -> Document:
        """
        Clean a single LangChain Document.
        
        Args:
            document: Document to clean.
            
        Returns:
            Cleaned Document with preserved metadata.
        """
        cleaned_text = self.clean_text(document.page_content)
        document.page_content = cleaned_text
        return document
    
    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean a list of LangChain Documents.
        
        Args:
            documents: List of Documents to clean.
            
        Returns:
            List of cleaned Documents.
        """
        logger.info(f"Cleaning {len(documents)} documents")
        
        cleaned_documents = []
        for doc in documents:
            cleaned_doc = self.clean_document(doc)
            cleaned_documents.append(cleaned_doc)
        
        logger.info(f"Cleaned {len(cleaned_documents)} documents")
        return cleaned_documents
