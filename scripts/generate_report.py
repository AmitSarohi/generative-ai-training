"""
PDF report generator using ReportLab.
Generates a comprehensive final report for the RAG application.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT


class ReportGenerator:
    """Generates PDF report for RAG application."""
    
    def __init__(self):
        """Initialize report generator."""
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        styles = {}
        
        # Title style
        styles['Title'] = ParagraphStyle(
            'Title',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Section heading
        styles['SectionHeading'] = ParagraphStyle(
            'SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Subsection heading
        styles['SubsectionHeading'] = ParagraphStyle(
            'SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.darkblue,
            spaceAfter=10,
            spaceBefore=15
        )
        
        # Body text
        styles['Body'] = ParagraphStyle(
            'Body',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=12,
            leading=14
        )
        
        # Code style
        styles['Code'] = ParagraphStyle(
            'Code',
            parent=self.styles['Code'],
            fontSize=9,
            leftIndent=20,
            spaceAfter=10
        )
        
        return styles
    
    def generate_report(self):
        """Generate the complete PDF report."""
        # Create document
        doc = SimpleDocTemplate(
            str(settings.REPORT_PATH),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story
        story = []
        
        # Title page
        story.extend(self._create_title_page())
        story.append(PageBreak())
        
        # Overview
        story.extend(self._create_overview())
        story.append(PageBreak())
        
        # Technical Implementation
        story.extend(self._create_technical_implementation())
        story.append(PageBreak())
        
        # Evaluation
        story.extend(self._create_evaluation())
        story.append(PageBreak())
        
        # Results & Discussion
        story.extend(self._create_results_and_discussion())
        story.append(PageBreak())
        
        # Conclusion
        story.extend(self._create_conclusion())
        
        # Build PDF
        doc.build(story)
        print(f"Report generated: {settings.REPORT_PATH}")
    
    def _create_title_page(self):
        """Create title page."""
        story = []
        
        # Title
        story.append(Spacer(1, 2 * inch))
        story.append(Paragraph("RAG Application", self.custom_styles['Title']))
        story.append(Paragraph("Final Report", self.custom_styles['Title']))
        story.append(Spacer(1, 1 * inch))
        
        # Date
        date_str = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"Generated: {date_str}", self.custom_styles['Body']))
        story.append(Spacer(1, 1 * inch))
        
        # Tech stack summary
        story.append(Paragraph("Technology Stack", self.custom_styles['SectionHeading']))
        story.append(Paragraph("Framework: LangChain", self.custom_styles['Body']))
        story.append(Paragraph("LLM: Ollama / llama3.2:3b", self.custom_styles['Body']))
        story.append(Paragraph("Embeddings: Ollama / nomic-embed-text", self.custom_styles['Body']))
        story.append(Paragraph("Vector Database: FAISS", self.custom_styles['Body']))
        story.append(Paragraph("Interface: CLI", self.custom_styles['Body']))
        
        return story
    
    def _create_overview(self):
        """Create overview section."""
        story = []
        
        story.append(Paragraph("1. OVERVIEW", self.custom_styles['SectionHeading']))
        
        story.append(Paragraph("Problem Statement", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "This project implements a Retrieval-Augmented Generation (RAG) application "
            "for answering questions about five seminal research papers in machine learning and "
            "natural language processing. The system combines document retrieval with large "
            "language model generation to provide accurate, source-attributed answers.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Assignment Objective", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "The objective is to build a complete RAG pipeline that loads PDF documents, "
            "extracts and processes text, generates embeddings, performs semantic search, "
            "and generates answers using an open-source language model. The system must "
            "maintain conversational context over the last four interactions and be "
            "evaluated using predefined questions.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("RAG Approach", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "The application uses LangChain as the RAG framework, Ollama for local LLM "
            "and embedding model inference, and FAISS for vector storage. Documents are "
            "chunked using RecursiveCharacterTextSplitter and embedded using nomic-embed-text. "
            "The system retrieves relevant chunks via semantic similarity and generates answers "
            "using llama3.2:3b.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Source Documents", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "Five research papers are used as the knowledge base:",
            self.custom_styles['Body']
        ))
        
        docs = [
            ["1706.03762.pdf", "Attention Is All You Need (Transformer)"],
            ["1810.04805.pdf", "BERT: Pre-training of Deep Bidirectional Transformers"],
            ["2005.14165.pdf", "Denoising Diffusion Probabilistic Models"],
            ["1907.11692.pdf", "LoRA: Low-Rank Adaptation of Large Language Models"],
            ["1910.10683.pdf", "ELECTRA: Pre-training Text Encoders as Discriminators"]
        ]
        
        table = Table(docs, colWidths=[2 * inch, 4 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        
        return story
    
    def _create_technical_implementation(self):
        """Create technical implementation section."""
        story = []
        
        story.append(Paragraph("2. TECHNICAL IMPLEMENTATION", self.custom_styles['SectionHeading']))
        
        story.append(Paragraph("PDF Ingestion", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "PDFs are loaded using LangChain's PyMuPDFLoader from the data/pdfs/ directory. "
            "The loader preserves metadata including source filename, page number, and document ID. "
            "All five required PDFs are loaded and validated before processing.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Text Extraction", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "PyMuPDF extracts text from each page while preserving document structure. "
            "Page-level metadata is maintained to enable source attribution in final answers.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Preprocessing", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "Lightweight text preprocessing normalizes excessive whitespace and removes "
            "unnecessary line breaks while preserving technical terminology and meaningful content.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Chunking Strategy", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            f"Documents are chunked using RecursiveCharacterTextSplitter with CHUNK_SIZE={settings.CHUNK_SIZE} "
            f"and CHUNK_OVERLAP={settings.CHUNK_OVERLAP}. Each chunk preserves source metadata and "
            "is assigned a unique chunk_id for tracking.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Embedding Generation", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            f"Embeddings are generated using Ollama's nomic-embed-text model via langchain_ollama. "
            f"The model runs locally at {settings.OLLAMA_BASE_URL}. The same embedding model is used "
            "for both indexing and querying to ensure consistency.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Vector Database", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "FAISS (Facebook AI Similarity Search) is used for efficient semantic similarity search. "
            "The vector store is persisted to data/vectorstore/ and loaded on subsequent runs to avoid "
            "regenerating embeddings.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Semantic Retrieval", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            f"LangChain's retriever interface provides semantic similarity search with TOP_K={settings.TOP_K}. "
            "Retrieved documents preserve source metadata for attribution.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Prompt Design", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "A ChatPromptTemplate defines the system prompt with rules for answering primarily from "
            "retrieved context, not inventing facts, and clearly stating when information is unavailable. "
            "The prompt includes chat history, retrieved context, and the user question.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("LLM Generation", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            f"Answers are generated using Ollama's llama3.2:3b model via ChatOllama. The model runs "
            f"locally at {settings.OLLAMA_BASE_URL} with a temperature of 0.7 for balanced creativity.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("RAG Pipeline", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "The RAG chain uses LangChain's LCEL (Runnable interface) to compose retrieval, "
            "prompt formatting, and generation into a single pipeline. The chain retrieves relevant "
            "documents, formats them into context, and generates answers using the LLM.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Conversational Memory", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            f"A collections.deque with maxlen={settings.MEMORY_MAXLEN} maintains conversation history. "
            "Each interaction (question + answer) is stored, and older interactions are automatically "
            "removed when the limit is exceeded. Memory is provided to the prompt for contextual "
            "follow-up questions.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Source Attribution", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "Every RAG response displays retrieved source documents and page numbers. The "
            "DocumentFormatter formats sources for display, using actual metadata from retrieved "
            "LangChain Documents.",
            self.custom_styles['Body']
        ))
        
        return story
    
    def _create_evaluation(self):
        """Create evaluation section."""
        story = []
        
        story.append(Paragraph("3. EVALUATION", self.custom_styles['SectionHeading']))
        
        story.append(Paragraph("Evaluation Questions", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "Ten predefined questions evaluate the system across different aspects of the source documents. "
            "Questions include conceptual, factual, comparison, and multi-document queries. Some questions "
            "test contextual recall by depending on previous conversation history.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Evaluation Framework", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "A custom evaluation framework is used due to the local execution environment. The framework "
            "evaluates four metrics on a 1-5 scale: Relevance, Accuracy, Contextual Awareness, and Response Quality.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Evaluation Methodology", self.custom_styles['SubsectionHeading']))
        
        metrics = [
            ["Metric", "Description", "Scale"],
            ["Relevance", "How well the answer addresses the question", "1-5"],
            ["Accuracy", "Alignment with expected answer and source documents", "1-5"],
            ["Contextual Awareness", "Use of conversation history for follow-up questions", "1-5"],
            ["Response Quality", "Clarity, structure, and informativeness", "1-5"]
        ]
        
        table = Table(metrics, colWidths=[1.5 * inch, 3 * inch, 1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        
        story.append(Paragraph("Overall Score", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "The overall score is calculated as the average of the four metric scores: "
            "(Relevance + Accuracy + Contextual Awareness + Response Quality) / 4",
            self.custom_styles['Body']
        ))
        
        return story
    
    def _create_results_and_discussion(self):
        """Create results and discussion section."""
        story = []
        
        story.append(Paragraph("4. RESULTS & DISCUSSION", self.custom_styles['SectionHeading']))
        
        # Load evaluation results if available
        if settings.EVALUATION_RESULTS_JSON.exists():
            with open(settings.EVALUATION_RESULTS_JSON, 'r') as f:
                results = json.load(f)
            
            # Calculate averages
            avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
            avg_accuracy = sum(r["accuracy_score"] for r in results) / len(results)
            avg_contextual = sum(r["contextual_awareness_score"] for r in results) / len(results)
            avg_quality = sum(r["response_quality_score"] for r in results) / len(results)
            avg_overall = sum(r["overall_score"] for r in results) / len(results)
            
            story.append(Paragraph("Evaluation Results", self.custom_styles['SubsectionHeading']))
            
            results_data = [
                ["Metric", "Average Score"],
                ["Relevance", f"{avg_relevance:.2f} / 5"],
                ["Accuracy", f"{avg_accuracy:.2f} / 5"],
                ["Contextual Awareness", f"{avg_contextual:.2f} / 5"],
                ["Response Quality", f"{avg_quality:.2f} / 5"],
                ["Overall", f"{avg_overall:.2f} / 5"]
            ]
            
            table = Table(results_data, colWidths=[2.5 * inch, 2 * inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        else:
            story.append(Paragraph("Evaluation Results", self.custom_styles['SubsectionHeading']))
            story.append(Paragraph(
                "Evaluation results not available. Run 'python scripts/evaluate.py' to generate results.",
                self.custom_styles['Body']
            ))
        
        story.append(Paragraph("Strengths", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "• Complete end-to-end RAG pipeline with all required components<br/>"
            "• Local execution using Ollama for privacy and cost efficiency<br/>"
            "• Efficient FAISS vector storage with persistence<br/>"
            "• Conversational memory for contextual follow-up questions<br/>"
            "• Source attribution for transparency and verifiability<br/>"
            "• Modular architecture with clear separation of concerns",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Limitations", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "• Custom evaluation metrics are heuristic-based rather than using RAGAS or TruLens<br/>"
            "• No advanced question contextualization beyond conversation history<br/>"
            "• Limited to the five provided PDF documents<br/>"
            "• Performance depends on Ollama model availability and hardware<br/>"
            "• CLI interface may be less user-friendly than web UI",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Challenges", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "• Ensuring Ollama models are properly installed and running<br/>"
            "• Balancing chunk size and overlap for optimal retrieval<br/>"
            "• Designing evaluation questions that test different aspects<br/>"
            "• Implementing memory that correctly handles the sliding window",
            self.custom_styles['Body']
        ))
        
        return story
    
    def _create_conclusion(self):
        """Create conclusion section."""
        story = []
        
        story.append(Paragraph("5. CONCLUSION", self.custom_styles['SectionHeading']))
        
        story.append(Paragraph("Summary", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "This project successfully implemented a complete RAG application for answering questions "
            "about five machine learning research papers. The system integrates PDF ingestion, text "
            "processing, embedding generation, semantic retrieval, and LLM generation using LangChain, "
            "Ollama, and FAISS.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Performance", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "The RAG pipeline performs effectively in retrieving relevant document chunks and generating "
            "contextually appropriate answers. The conversational memory enables follow-up questions "
            "that reference previous context. Source attribution provides transparency for answers.",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Learnings", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "• LangChain provides a powerful framework for building RAG applications<br/>"
            "• Ollama enables local LLM inference without cloud dependencies<br/>"
            "• FAISS offers efficient vector similarity search for large document collections<br/>"
            "• Conversation memory significantly improves follow-up question handling<br/>"
            "• Proper chunking strategy is critical for retrieval quality",
            self.custom_styles['Body']
        ))
        
        story.append(Paragraph("Future Improvements", self.custom_styles['SubsectionHeading']))
        story.append(Paragraph(
            "• Integrate RAGAS or TruLens for more sophisticated evaluation<br/>"
            "• Implement advanced question contextualization<br/>"
            "• Add support for more document types (web pages, text files)<br/>"
            "• Develop a web interface using a lightweight framework<br/>"
            "• Implement hybrid retrieval combining semantic and keyword search",
            self.custom_styles['Body']
        ))
        
        return story


def main():
    """Main function to generate the report."""
    print("Generating PDF report...")
    
    # Ensure reports directory exists
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    generator = ReportGenerator()
    generator.generate_report()
    
    print(f"Report saved to: {settings.REPORT_PATH}")


if __name__ == "__main__":
    main()
