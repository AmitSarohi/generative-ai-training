"""
FAISS vector store module using LangChain's FAISS integration.
Creates, persists, and loads FAISS vector stores for semantic search.
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Manages FAISS vector store for document embeddings."""
    
    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        vectorstore_path: Path = None
    ):
        """
        Initialize FAISS vector store manager.
        
        Args:
            embeddings: OllamaEmbeddings instance.
            vectorstore_path: Path to store/load vector store. Defaults to settings.VECTORSTORE_PATH.
        """
        self.embeddings = embeddings
        self.vectorstore_path = vectorstore_path or settings.VECTORSTORE_PATH
        self.vectorstore: Optional[FAISS] = None
        
        # Ensure vectorstore directory exists
        self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of Document objects with metadata.
            
        Returns:
            FAISS vector store instance.
        """
        try:
            logger.info(f"Creating FAISS vector store from {len(documents)} documents")
            
            # Extract texts from documents
            texts = [doc.page_content for doc in documents]
            
            # Create FAISS vector store
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            logger.info("FAISS vector store created successfully")
            return self.vectorstore
            
        except Exception as e:
            error_msg = f"Failed to create FAISS vector store: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def save_vectorstore(self) -> None:
        """
        Persist FAISS vector store to disk.
        """
        if self.vectorstore is None:
            error_msg = "No vector store to save. Create one first."
            logger.error(error_msg)
            raise Exception(error_msg)
        
        try:
            logger.info(f"Saving vector store to {self.vectorstore_path}")
            self.vectorstore.save_local(str(self.vectorstore_path))
            logger.info("Vector store saved successfully")
        except Exception as e:
            error_msg = f"Failed to save vector store: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def load_vectorstore(self) -> FAISS:
        """
        Load FAISS vector store from disk.
        
        Returns:
            Loaded FAISS vector store instance.
            
        Note:
            Uses allow_dangerous_deserialization=True for locally generated trusted vector store.
        """
        try:
            logger.info(f"Loading vector store from {self.vectorstore_path}")
            
            self.vectorstore = FAISS.load_local(
                str(self.vectorstore_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info("Vector store loaded successfully")
            return self.vectorstore
            
        except Exception as e:
            error_msg = f"Failed to load vector store: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query string.
            k: Number of results to return. Defaults to settings.TOP_K.
            
        Returns:
            List of retrieved Document objects.
        """
        if self.vectorstore is None:
            error_msg = "No vector store available. Create or load one first."
            logger.error(error_msg)
            raise Exception(error_msg)
        
        if k is None:
            k = settings.TOP_K
        
        logger.info(f"Performing similarity search with k={k}")
        documents = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(documents)} documents")
        
        return documents
    
    def get_retriever(self, search_kwargs: dict = None):
        """
        Get a retriever from the vector store.
        
        Args:
            search_kwargs: Search parameters like k for top-k results.
            
        Returns:
            LangChain retriever instance.
        """
        if self.vectorstore is None:
            error_msg = "No vector store available. Create or load one first."
            logger.error(error_msg)
            raise Exception(error_msg)
        
        if search_kwargs is None:
            search_kwargs = {"k": settings.TOP_K}
        
        logger.info(f"Creating retriever with search_kwargs: {search_kwargs}")
        retriever = self.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )
        
        return retriever
    
    def exists(self) -> bool:
        """
        Check if a persisted vector store exists.
        
        Returns:
            True if vector store files exist, False otherwise.
        """
        index_path = self.vectorstore_path / "index.faiss"
        return index_path.exists()
