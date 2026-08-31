"""
Document formatter module for formatting retrieved documents into prompt context.
Formats retrieved documents with source metadata for attribution.
"""

from typing import List

from langchain_core.documents import Document


class DocumentFormatter:
    """Formats retrieved documents for prompt context."""
    
    def format_documents(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a string for the prompt.
        
        Args:
            documents: List of retrieved Document objects.
            
        Returns:
            Formatted string with document content and metadata.
        """
        if not documents:
            return "No relevant documents were retrieved."
        
        formatted_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            content = doc.page_content
            
            formatted_part = f"[Source {i}]\nDocument: {source}\nPage: {page}\n\nContent:\n{content}\n"
            formatted_parts.append(formatted_part)
        
        return "\n".join(formatted_parts)
    
    def format_sources(self, documents: List[Document]) -> List[str]:
        """
        Format source information for display.
        
        Args:
            documents: List of retrieved Document objects.
            
        Returns:
            List of formatted source strings.
        """
        sources = []
        seen = set()
        
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            
            # Create unique key to avoid duplicates
            key = f"{source}_page_{page}"
            if key not in seen:
                seen.add(key)
                sources.append(f"{source} - Page {page}")
        
        return sources
