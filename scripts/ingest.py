"""
Ingestion script for processing PDFs and creating the vector store.
Loads PDFs, extracts text, preprocesses, chunks, embeds, and stores in FAISS.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.ingestion.loader import PDFLoader
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import DocumentChunker
from app.embeddings.embeddings import EmbeddingGenerator
from app.vectorstore.faiss_store import FAISSVectorStore


def main():
    """Main ingestion pipeline."""
    print("=" * 50)
    print("RAG INGESTION PIPELINE")
    print("=" * 50)
    print()
    
    try:
        # Step 1: Verify PDFs exist
        print("Step 1: Verifying PDFs...")
        pdf_loader = PDFLoader()
        
        if not pdf_loader.verify_pdfs_exist():
            print("Error: Not all required PDFs are present.")
            print()
            print("Please ensure the following PDFs are in data/pdfs/:")
            for pdf in settings.REQUIRED_PDFS:
                print(f"  - {pdf}")
            print()
            print("Download from:")
            print("  https://arxiv.org/pdf/1706.03762.pdf")
            print("  https://arxiv.org/pdf/1810.04805.pdf")
            print("  https://arxiv.org/pdf/2005.14165.pdf")
            print("  https://arxiv.org/pdf/1907.11692.pdf")
            print("  https://arxiv.org/pdf/1910.10683.pdf")
            print()
            return
        
        print("All required PDFs are present.")
        print()
        
        # Step 2: Load PDFs
        print("Step 2: Loading PDFs...")
        documents = pdf_loader.load_all_pdfs()
        print(f"PDFs processed: {len(settings.REQUIRED_PDFS)}")
        print(f"Pages processed: {len(documents)}")
        print()
        
        # Step 3: Clean text
        print("Step 3: Cleaning text...")
        text_cleaner = TextCleaner()
        cleaned_documents = text_cleaner.clean_documents(documents)
        print()
        
        # Step 4: Chunk documents
        print("Step 4: Chunking documents...")
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(cleaned_documents)
        print(f"Chunks created: {len(chunks)}")
        print()
        
        # Step 5: Initialize embeddings
        print("Step 5: Initializing embeddings...")
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.get_embeddings()
        print(f"Embedding model: {settings.OLLAMA_EMBEDDING_MODEL}")
        print()
        
        # Step 6: Create vector store
        print("Step 6: Creating FAISS vector store...")
        vectorstore_manager = FAISSVectorStore(embeddings)
        vectorstore = vectorstore_manager.create_vectorstore(chunks)
        print(f"Vector database: FAISS")
        print()
        
        # Step 7: Persist vector store
        print("Step 7: Persisting vector store...")
        vectorstore_manager.save_vectorstore()
        print(f"Vector store: {settings.VECTORSTORE_DIR}")
        print()
        
        # Step 8: Verify vector store can be loaded
        print("Step 8: Verifying vector store can be loaded...")
        test_vectorstore_manager = FAISSVectorStore(embeddings)
        test_vectorstore = test_vectorstore_manager.load_vectorstore()
        print("Vector store loaded successfully.")
        print()
        
        print("=" * 50)
        print("INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print()
        print("Summary:")
        print(f"  PDFs processed: {len(settings.REQUIRED_PDFS)}")
        print(f"  Pages processed: {len(documents)}")
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Embedding model: {settings.OLLAMA_EMBEDDING_MODEL}")
        print(f"  Vector database: FAISS")
        print(f"  Vector store: {settings.VECTORSTORE_DIR}")
        print()
        print("You can now run the chatbot:")
        print("  python scripts/chat.py")
        print()
    
    except Exception as e:
        print()
        print(f"Error during ingestion: {str(e)}")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
