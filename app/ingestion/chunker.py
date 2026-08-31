"""
Text chunking module using LangChain's RecursiveCharacterTextSplitter.
Splits documents into meaningful chunks with overlap.
"""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunks documents using RecursiveCharacterTextSplitter."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk. Defaults to settings.CHUNK_SIZE.
            chunk_overlap: Overlap between chunks. Defaults to settings.CHUNK_OVERLAP.
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Documents to chunk.
            
        Returns:
            List of chunked Documents with added chunk_id metadata.
        """
        logger.info(f"Chunking {len(documents)} documents")
        logger.info(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk_id to each chunk's metadata
        chunk_counter = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            document_id = chunk.metadata.get("document_id", "unknown")
            
            # Initialize counter for this document
            if document_id not in chunk_counter:
                chunk_counter[document_id] = {}
            if page not in chunk_counter[document_id]:
                chunk_counter[document_id][page] = 0
            
            chunk_num = chunk_counter[document_id][page] + 1
            chunk_counter[document_id][page] = chunk_num
            
            chunk.metadata["chunk_id"] = f"{document_id}_page_{page}_chunk_{chunk_num}"
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
