"""
Configuration module for RAG application.
Uses python-dotenv for environment variable management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings and configuration."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    PDFS_DIR = DATA_DIR / "pdfs"
    VECTORSTORE_DIR = DATA_DIR / "vectorstore"
    REPORTS_DIR = BASE_DIR / "reports"
    EVALUATION_DIR = BASE_DIR / "evaluation"
    
    # Required PDF filenames
    REQUIRED_PDFS = [
        "1706.03762.pdf",
        "1810.04805.pdf",
        "2005.14165.pdf",
        "1907.11692.pdf",
        "1910.10683.pdf"
    ]
    
    # Ollama configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # Chunking configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    
    # Retrieval configuration
    TOP_K = int(os.getenv("TOP_K", "5"))
    
    # Memory configuration
    MEMORY_MAXLEN = 4
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Vector store
    VECTORSTORE_FILENAME = "faiss_index"
    VECTORSTORE_PATH = VECTORSTORE_DIR / VECTORSTORE_FILENAME
    
    # Evaluation
    EVALUATION_QUESTIONS_FILE = EVALUATION_DIR / "questions.json"
    EVALUATION_RESULTS_JSON = EVALUATION_DIR / "results.json"
    EVALUATION_RESULTS_CSV = EVALUATION_DIR / "results.csv"
    
    # Report
    REPORT_FILENAME = "final_report.pdf"
    REPORT_PATH = REPORTS_DIR / REPORT_FILENAME


# Global settings instance
settings = Settings()
