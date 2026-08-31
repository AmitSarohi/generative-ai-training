"""
Ollama embeddings module using langchain_ollama.
Uses nomic-embed-text model for generating embeddings.
"""

import logging
from typing import List

from langchain_ollama import OllamaEmbeddings

from app.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using Ollama's nomic-embed-text model."""
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            model: Embedding model name. Defaults to settings.OLLAMA_EMBEDDING_MODEL.
            base_url: Ollama base URL. Defaults to settings.OLLAMA_BASE_URL.
        """
        self.model = model or settings.OLLAMA_EMBEDDING_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        
        logger.info(f"Initializing OllamaEmbeddings with model: {self.model}")
        logger.info(f"Ollama base URL: {self.base_url}")
        
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url
            )
            logger.info("OllamaEmbeddings initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize OllamaEmbeddings: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_embeddings(self) -> OllamaEmbeddings:
        """
        Get the OllamaEmbeddings instance.
        
        Returns:
            OllamaEmbeddings instance.
        """
        return self.embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        try:
            logger.info(f"Embedding {len(texts)} documents")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully embedded {len(embeddings)} documents")
            return embeddings
        except Exception as e:
            error_msg = f"Failed to embed documents: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed.
            
        Returns:
            Embedding vector.
        """
        try:
            logger.info("Embedding query")
            embedding = self.embeddings.embed_query(text)
            logger.info("Successfully embedded query")
            return embedding
        except Exception as e:
            error_msg = f"Failed to embed query: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
