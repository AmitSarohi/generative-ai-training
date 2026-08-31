"""
RAG chain module using LangChain's LCEL (Runnable interface).
Integrates retrieval, memory, and generation into a single pipeline.
"""

import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.rag.prompts import get_rag_prompt
from app.rag.document_formatter import DocumentFormatter
from app.retrieval.retriever import SemanticRetriever
from app.llm.ollama import OllamaLLM

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain using LangChain LCEL."""
    
    def __init__(
        self,
        retriever: SemanticRetriever,
        llm: OllamaLLM
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: SemanticRetriever instance.
            llm: OllamaLLM instance.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt = get_rag_prompt()
        self.document_formatter = DocumentFormatter()
        self.output_parser = StrOutputParser()
        
        # Build the chain
        self.chain = self._build_chain()
        
        logger.info("RAG chain initialized successfully")
    
    def _build_chain(self):
        """
        Build the RAG chain using LCEL.
        
        Returns:
            LangChain Runnable chain.
        """
        # Build the chain (context is pre-formatted in invoke method)
        chain = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | self.prompt
            | self.llm.get_llm()
            | self.output_parser
        )
        
        return chain
    
    def invoke(
        self,
        question: str,
        chat_history: str = ""
    ) -> Tuple[str, List[Document]]:
        """
        Invoke the RAG chain with a question.
        
        Args:
            question: User question.
            chat_history: Conversation history string.
            
        Returns:
            Tuple of (answer, retrieved_documents).
        """
        try:
            logger.info(f"Invoking RAG chain with question: {question[:100]}...")
            
            # Retrieve documents separately for source attribution
            retrieved_docs = self.retriever.retrieve(question)
            
            # Format context for the prompt
            context = self.document_formatter.format_documents(retrieved_docs)
            
            # Format the prompt directly
            prompt_value = self.prompt.format(
                context=context,
                question=question,
                chat_history=chat_history
            )
            
            # Invoke the LLM directly
            logger.info("Invoking LLM...")
            answer = self.llm.get_llm().invoke(prompt_value)
            
            # Parse the output - convert AIMessage to string
            answer = self.output_parser.parse(answer)
            
            # Ensure answer is a string
            if hasattr(answer, 'content'):
                answer = answer.content
            
            logger.info("RAG chain invocation completed")
            
            return answer, retrieved_docs
            
        except Exception as e:
            error_msg = f"Failed to invoke RAG chain: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
