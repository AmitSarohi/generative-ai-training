"""
Unit tests for evaluation questions.
"""

import pytest
import json
from pathlib import Path

from app.config import settings


def test_questions_file_exists():
    """Test that questions.json exists."""
    assert settings.EVALUATION_QUESTIONS_FILE.exists()


def test_questions_count():
    """Test that there are exactly 10 questions."""
    with open(settings.EVALUATION_QUESTIONS_FILE, 'r') as f:
        questions = json.load(f)
    
    assert len(questions) == 10


def test_question_structure():
    """Test that each question has required fields."""
    with open(settings.EVALUATION_QUESTIONS_FILE, 'r') as f:
        questions = json.load(f)
    
    required_fields = ["id", "question", "expected_answer", "source_documents", "tests_memory"]
    
    for q in questions:
        for field in required_fields:
            assert field in q, f"Question missing field: {field}"


def test_question_ids_unique():
    """Test that question IDs are unique."""
    with open(settings.EVALUATION_QUESTIONS_FILE, 'r') as f:
        questions = json.load(f)
    
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "Question IDs are not unique"
