"""
Semantic retrieval module using LangChain retriever.
Provides semantic similarity search over the vector store.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Manages semantic retrieval using LangChain retriever."""
    
    def __init__(self, retriever: BaseRetriever):
        """
        Initialize semantic retriever.
        
        Args:
            retriever: LangChain retriever instance from vector store.
        """
        self.retriever = retriever
        self.top_k = settings.TOP_K
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string.
            
        Returns:
            List of retrieved Document objects with metadata.
        """
        try:
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            
            # Use the retriever's invoke method
            documents = self.retriever.invoke(query)
            
            logger.info(f"Retrieved {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            error_msg = f"Failed to retrieve documents: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_top_k(self) -> int:
        """
        Get the current top-k value.
        
        Returns:
            Number of documents to retrieve.
        """
        return self.top_k
