"""
Ollama LLM module using langchain_ollama.
Uses llama3.2:3b model for answer generation.
"""

import logging

from langchain_ollama import ChatOllama

from app.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class OllamaLLM:
    """Manages Ollama LLM for answer generation."""
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.7
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model: Model name. Defaults to settings.OLLAMA_MODEL.
            base_url: Ollama base URL. Defaults to settings.OLLAMA_BASE_URL.
            temperature: Generation temperature.
        """
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.temperature = temperature
        
        logger.info(f"Initializing ChatOllama with model: {self.model}")
        logger.info(f"Ollama base URL: {self.base_url}")
        
        try:
            self.llm = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                timeout=60  # 60 second timeout
            )
            logger.info("ChatOllama initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize ChatOllama: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_llm(self) -> ChatOllama:
        """
        Get the ChatOllama instance.
        
        Returns:
            ChatOllama instance.
        """
        return self.llm
