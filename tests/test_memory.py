"""
Unit tests for conversation memory module.
"""

import pytest

from app.memory.conversation_memory import ConversationMemory


def test_memory_initialization():
    """Test memory initialization."""
    memory = ConversationMemory()
    assert memory.maxlen == 4
    assert memory.is_empty()


def test_add_interaction():
    """Test adding interactions to memory."""
    memory = ConversationMemory()
    
    memory.add_interaction("Question 1", "Answer 1")
    assert memory.count() == 1
    assert not memory.is_empty()


def test_memory_sliding_window():
    """Test that memory respects maxlen (sliding window)."""
    memory = ConversationMemory(maxlen=4)
    
    # Add 5 interactions
    for i in range(1, 6):
        memory.add_interaction(f"Question {i}", f"Answer {i}")
    
    # Should only have 4 interactions
    assert memory.count() == 4
    
    # First interaction should be removed
    history = memory.get_messages()
    assert len(history) == 4
    assert history[0][0] == "Question 2"  # Question 1 was removed
    assert history[-1][0] == "Question 5"


def test_get_history():
    """Test getting conversation history."""
    memory = ConversationMemory()
    
    memory.add_interaction("Q1", "A1")
    memory.add_interaction("Q2", "A2")
    
    history = memory.get_history()
    assert "Q1" in history
    assert "A1" in history
    assert "Q2" in history
    assert "A2" in history


def test_get_messages():
    """Test getting messages as tuples."""
    memory = ConversationMemory()
    
    memory.add_interaction("Q1", "A1")
    memory.add_interaction("Q2", "A2")
    
    messages = memory.get_messages()
    assert len(messages) == 2
    assert messages[0] == ("Q1", "A1")
    assert messages[1] == ("Q2", "A2")


def test_clear():
    """Test clearing memory."""
    memory = ConversationMemory()
    
    memory.add_interaction("Q1", "A1")
    memory.add_interaction("Q2", "A2")
    assert memory.count() == 2
    
    memory.clear()
    assert memory.count() == 0
    assert memory.is_empty()


def test_count():
    """Test counting interactions."""
    memory = ConversationMemory()
    
    assert memory.count() == 0
    
    memory.add_interaction("Q1", "A1")
    assert memory.count() == 1
    
    memory.add_interaction("Q2", "A2")
    assert memory.count() == 2


def test_is_empty():
    """Test is_empty check."""
    memory = ConversationMemory()
    
    assert memory.is_empty()
    
    memory.add_interaction("Q1", "A1")
    assert not memory.is_empty()
