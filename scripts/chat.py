"""
CLI chatbot for RAG application.
Provides interactive command-line interface for querying the RAG system.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.embeddings.embeddings import EmbeddingGenerator
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.retriever import SemanticRetriever
from app.llm.ollama import OllamaLLM
from app.rag.rag_chain import RAGChain
from app.memory.conversation_memory import ConversationMemory
from app.rag.document_formatter import DocumentFormatter

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


def print_header():
    """Print the chatbot header."""
    print("=" * 50)
    print("RAG RESEARCH ASSISTANT")
    print("=" * 50)
    print()
    print("Framework:")
    print("LangChain")
    print()
    print("LLM:")
    print("Ollama / llama3.2:3b")
    print()
    print("Embeddings:")
    print("Ollama / nomic-embed-text")
    print()
    print("Vector Database:")
    print("FAISS")
    print()
    print("Top K:")
    print(settings.TOP_K)
    print()
    print("Conversation Memory:")
    print(f"Last {settings.MEMORY_MAXLEN} interactions")
    print()
    print("=" * 50)
    print()


def print_help():
    """Print available commands."""
    print()
    print("Commands:")
    print("  exit / quit  - Exit the application")
    print("  clear        - Clear conversation memory")
    print("  memory       - Display current conversation memory")
    print("  help         - Display available commands")
    print()


def print_memory(memory: ConversationMemory):
    """Print current conversation memory."""
    print()
    print("=" * 50)
    print("CONVERSATION MEMORY")
    print("=" * 50)
    
    if memory.is_empty():
        print("Memory is empty.")
    else:
        history = memory.get_history()
        print(history)
    
    print("=" * 50)
    print()


def main():
    """Main chatbot loop."""
    print_header()
    print_help()
    
    try:
        # Initialize components
        logger.info("Initializing RAG components...")
        
        # Initialize embeddings
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.get_embeddings()
        
        # Load vector store
        vectorstore_manager = FAISSVectorStore(embeddings)
        
        if not vectorstore_manager.exists():
            print("Error: Vector store not found.")
            print("Please run the ingestion script first:")
            print("  python scripts/ingest.py")
            print()
            return
        
        vectorstore_manager.load_vectorstore()
        
        # Initialize retriever
        retriever = SemanticRetriever(vectorstore_manager.get_retriever())
        
        # Initialize LLM
        llm = OllamaLLM()
        
        # Initialize RAG chain
        rag_chain = RAGChain(retriever, llm)
        
        # Initialize memory
        memory = ConversationMemory()
        
        # Initialize document formatter
        document_formatter = DocumentFormatter()
        
        logger.info("RAG components initialized successfully")
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input("User: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ["exit", "quit"]:
                    print()
                    print("Goodbye!")
                    print()
                    break
                
                if user_input.lower() == "clear":
                    memory.clear()
                    print("Conversation memory cleared.")
                    print()
                    continue
                
                if user_input.lower() == "memory":
                    print_memory(memory)
                    continue
                
                if user_input.lower() == "help":
                    print_help()
                    continue
                
                # Process question
                print()
                print("Assistant:", end=" ")
                
                # Get conversation history
                chat_history = memory.get_history()
                
                # Invoke RAG chain
                answer, retrieved_docs = rag_chain.invoke(
                    question=user_input,
                    chat_history=chat_history
                )
                
                print(answer)
                print()
                
                # Print sources
                sources = document_formatter.format_sources(retrieved_docs)
                print("Sources:")
                for source in sources:
                    print(f"- {source}")
                print()
                
                # Print memory status
                print(f"Memory: {memory.count()}/{settings.MEMORY_MAXLEN}")
                print()
                
                # Add interaction to memory
                memory.add_interaction(user_input, answer)
                
            except KeyboardInterrupt:
                print()
                print()
                print("Goodbye!")
                print()
                break
            except Exception as e:
                print()
                print(f"Error: {str(e)}")
                print()
                logger.error(f"Error in chat loop: {str(e)}")
    
    except Exception as e:
        print()
        print(f"Initialization error: {str(e)}")
        print()
        logger.error(f"Initialization error: {str(e)}")


if __name__ == "__main__":
    main()
