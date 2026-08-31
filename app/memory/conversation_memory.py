"""
Conversation memory module using collections.deque.
Maintains conversational context over the last four interactions.
"""

import logging
from collections import deque
from typing import List, Tuple

from app.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation memory with a sliding window of 4 interactions."""
    
    def __init__(self, maxlen: int = None):
        """
        Initialize conversation memory.
        
        Args:
            maxlen: Maximum number of interactions to store. Defaults to settings.MEMORY_MAXLEN (4).
        """
        self.maxlen = maxlen or settings.MEMORY_MAXLEN
        self.memory = deque(maxlen=self.maxlen)
        logger.info(f"Conversation memory initialized with maxlen={self.maxlen}")
    
    def add_interaction(self, question: str, answer: str) -> None:
        """
        Add a question-answer interaction to memory.
        
        Args:
            question: User's question.
            answer: Assistant's answer.
        """
        interaction = {"question": question, "answer": answer}
        self.memory.append(interaction)
        logger.info(f"Added interaction to memory. Current count: {len(self.memory)}")
    
    def get_history(self) -> str:
        """
        Get conversation history as a formatted string.
        
        Returns:
            Formatted conversation history string.
        """
        if not self.memory:
            return ""
        
        history_parts = []
        for i, interaction in enumerate(self.memory, 1):
            history_parts.append(f"Turn {i}:")
            history_parts.append(f"User: {interaction['question']}")
            history_parts.append(f"Assistant: {interaction['answer']}")
            history_parts.append("")
        
        return "\n".join(history_parts)
    
    def get_messages(self) -> List[Tuple[str, str]]:
        """
        Get conversation history as a list of (question, answer) tuples.
        
        Returns:
            List of (question, answer) tuples.
        """
        return [(interaction["question"], interaction["answer"]) for interaction in self.memory]
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def count(self) -> int:
        """
        Get the current number of interactions in memory.
        
        Returns:
            Number of interactions stored.
        """
        return len(self.memory)
    
    def is_empty(self) -> bool:
        """
        Check if memory is empty.
        
        Returns:
            True if memory is empty, False otherwise.
        """
        return len(self.memory) == 0
