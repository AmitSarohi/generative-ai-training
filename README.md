# RAG Research Assistant

A complete Retrieval-Augmented Generation (RAG) application for answering questions about machine learning research papers using LangChain, Ollama, and FAISS.

## Project Overview

This project implements a RAG pipeline that:
- Loads five research papers from PDF files
- Extracts, preprocesses, and chunks text
- Generates embeddings using Ollama's nomic-embed-text
- Stores embeddings in FAISS vector database
- Retrieves relevant chunks via semantic similarity
- Generates answers using Ollama's llama3.2:3b
- Maintains conversation context over the last four interactions
- Evaluates system performance with predefined questions
- Generates a comprehensive PDF report

## Assignment Requirements

The application fulfills the following requirements:
- Load five specified PDF documents manually placed in data/pdfs/
- Extract and preprocess text from PDFs
- Split text into meaningful chunks with overlap
- Generate embeddings using Ollama (nomic-embed-text)
- Store embeddings in FAISS vector database
- Configure efficient semantic similarity search
- Retrieve relevant chunks for user questions
- Generate answers using retrieved context and user queries
- Maintain conversational context over last four interactions
- Evaluate system using ten predefined questions
- Evaluate Relevance, Accuracy, Contextual Awareness, and Response Quality
- Produce final PDF report
- Provide source code with comments and documentation
- Provide configuration and environment setup instructions

## Architecture

```
5 LOCAL PDFs
      |
      v
PyMuPDFLoader
      |
      v
LangChain Documents
      |
      v
Text Preprocessing
      |
      v
RecursiveCharacterTextSplitter
      |
      v
Document Chunks
      |
      v
OllamaEmbeddings
      |
      v
nomic-embed-text
      |
      v
FAISS Vector Store
      |
      |
      User Question
      |
      v
LangChain Retriever
      |
      v
Retrieved Context
      |
+------------+------------+
|                         |
v                         v
Retrieved Documents    Last 4 Interactions
|                         |
+------------+------------+
      |
      v
ChatPromptTemplate
      |
      v
ChatOllama
      |
      v
llama3.2:3b
      |
      v
Answer
      |
      v
Update Memory
```

## Technology Stack

### RAG Framework
- **LangChain**: Framework for building RAG applications with LCEL (LangChain Expression Language)

### PDF Processing
- **PyMuPDF**: PDF text extraction via langchain-community's PyMuPDFLoader

### Text Splitting
- **RecursiveCharacterTextSplitter**: Intelligent text chunking with configurable size and overlap

### Embeddings
- **OllamaEmbeddings**: Local embedding generation via langchain-ollama
- **nomic-embed-text**: Embedding model for semantic search

### Vector Database
- **FAISS**: Efficient similarity search and clustering of dense vectors (faiss-cpu)

### LLM
- **ChatOllama**: Local LLM inference via langchain-ollama
- **llama3.2:3b**: Generation model for answer production

### Configuration
- **python-dotenv**: Environment variable management

### Evaluation
- **Custom evaluation framework**: Heuristic-based metrics for Relevance, Accuracy, Contextual Awareness, and Response Quality

### Report Generation
- **ReportLab**: PDF report generation

### Testing
- **pytest**: Unit testing framework

### Interface
- **CLI**: Command-line interface for interactive querying

## Why LangChain?

LangChain provides:
- Unified interface for different LLM providers
- Built-in document loaders and text splitters
- LCEL for composable, type-safe chains
- Easy integration with vector stores
- Prompt template management
- Memory abstractions for conversation context

## Why Ollama?

Ollama provides:
- Local LLM inference without cloud dependencies
- Privacy and security (data stays local)
- Cost efficiency (no API fees)
- Support for multiple models (llama3.2:3b, nomic-embed-text)
- Easy installation and model management
- HTTP API for integration

## Why llama3.2:3b?

llama3.2:3b provides:
- Good balance of performance and resource usage
- Suitable for local execution on consumer hardware
- Strong language understanding and generation
- Active development and support from Meta

## Why nomic-embed-text?

nomic-embed-text provides:
- High-quality embeddings for semantic search
- Efficient inference on local hardware
- Good performance on technical documents
- Compatibility with Ollama's infrastructure

## Why FAISS?

FAISS provides:
- Efficient similarity search at scale
- CPU-optimized implementation (faiss-cpu)
- Fast indexing and retrieval
- Persistence support for vector stores
- Widely used and well-tested

## Five Required PDFs

The application requires these five research papers:

1. **1706.03762.pdf** - Attention Is All You Need (Transformer)
   - https://arxiv.org/pdf/1706.03762.pdf

2. **1810.04805.pdf** - BERT: Pre-training of Deep Bidirectional Transformers
   - https://arxiv.org/pdf/1810.04805.pdf

3. **2005.14165.pdf** - Denoising Diffusion Probabilistic Models
   - https://arxiv.org/pdf/2005.14165.pdf

4. **1907.11692.pdf** - LoRA: Low-Rank Adaptation of Large Language Models
   - https://arxiv.org/pdf/1907.11692.pdf

5. **1910.10683.pdf** - ELECTRA: Pre-training Text Encoders as Discriminators
   - https://arxiv.org/pdf/1910.10683.pdf

## Manual PDF Placement

The application does NOT download PDFs automatically. You must manually download and place them:

```
data/pdfs/1706.03762.pdf
data/pdfs/1810.04805.pdf
data/pdfs/2005.14165.pdf
data/pdfs/1907.11692.pdf
data/pdfs/1910.10683.pdf
```

## PDF Ingestion

The ingestion pipeline:
1. Verifies all five PDFs exist in data/pdfs/
2. Loads PDFs using PyMuPDFLoader
3. Extracts text with page-level metadata
4. Preserves source, filename, page, and document_id metadata
5. Handles missing or unreadable PDFs gracefully

## Text Extraction

PyMuPDF through LangChain's PyMuPDFLoader:
- Extracts text from each page
- Preserves document structure where possible
- Maintains page numbers for source attribution
- Handles tables and figures where practical

## Preprocessing

Lightweight text preprocessing:
- Normalizes excessive whitespace
- Removes unnecessary line breaks
- Preserves technical terminology
- Maintains meaningful content
- Does not aggressively transform source material

## Chunking

Chunking strategy:
- **CHUNK_SIZE**: 800 characters (configurable)
- **CHUNK_OVERLAP**: 120 characters (configurable)
- **Splitter**: RecursiveCharacterTextSplitter
- **Separators**: ["\n\n", "\n", ". ", " ", ""]
- **Metadata preserved**: source, filename, page, document_id, chunk_id

Each chunk receives a unique chunk_id for tracking:
```
{document_id}_page_{page}_chunk_{chunk_number}
```

## Embeddings

Embedding generation:
- **Model**: nomic-embed-text via Ollama
- **Base URL**: http://localhost:11434
- **Framework**: langchain-ollama
- **Same model** used for indexing and querying
- Handles Ollama connection errors gracefully

## Vector Database

FAISS vector store:
- **Location**: data/vectorstore/
- **Persistence**: Saved after ingestion, loaded on chat startup
- **Deserialization**: allow_dangerous_deserialization=True for locally generated trusted store
- **No regeneration** of embeddings on each chat start

## Retrieval

Semantic retrieval:
- **Search type**: similarity
- **TOP_K**: 5 (configurable)
- **Retriever**: LangChain retriever interface
- **Metadata preserved**: source, filename, page, chunk_id
- Source attribution from actual retrieved documents

## RAG Generation

RAG pipeline using LCEL:
1. User question → LangChain Retriever
2. Retrieved documents → Document Formatter
3. Formatted context + Chat History → ChatPromptTemplate
4. Prompt → ChatOllama (llama3.2:3b)
5. Output → StrOutputParser
6. Answer → Update Memory

## Conversation Memory

Memory implementation:
- **Storage**: collections.deque(maxlen=4)
- **Content**: Last 4 interaction pairs (question + answer)
- **Sliding window**: Oldest interaction removed when limit exceeded
- **Methods**: add_interaction(), get_history(), get_messages(), clear(), count(), is_empty()

### Memory Behavior Example

```
Turn 1: User: What is attention? → Assistant: ...
Turn 2: User: How does it work? → Assistant: ...
Turn 3: User: Why is it useful? → Assistant: ...
Turn 4: User: What are its limitations? → Assistant: ...
Turn 5: User: Compare it with RNNs.

After Turn 5:
- Turn 1 is removed
- Memory contains: Turn 2, Turn 3, Turn 4, Turn 5
```

## CLI Usage

### Start Chatbot

```bash
python scripts/chat.py
```

### Commands

- `exit` / `quit`: Exit the application
- `clear`: Clear conversation memory
- `memory`: Display current conversation memory
- `help`: Display available commands

### Example Session

```
User: What is the Transformer architecture?
Assistant: The Transformer architecture is a neural network architecture...
Sources:
- 1706.03762.pdf - Page 3
- 1706.03762.pdf - Page 4
Memory: 1/4

User: What are its advantages?
Assistant: The Transformer offers several advantages...
Sources:
- 1706.03762.pdf - Page 5
Memory: 2/4
```

## Evaluation

### Evaluation Questions

Ten predefined questions in `evaluation/questions.json`:
- Grounded in the five PDFs
- Cover different aspects of PDF content
- Test retrieval and contextual recall
- Include conceptual, factual, comparison, and multi-document questions
- Mark questions that test memory with `"tests_memory": true`

### Evaluation Framework

Custom evaluation framework with four metrics (1-5 scale):

- **Relevance**: How well the answer addresses the question
- **Accuracy**: Alignment with expected answer and source documents
- **Contextual Awareness**: Use of conversation history for follow-up questions
- **Response Quality**: Clarity, structure, and informativeness

**Overall Score**: (Relevance + Accuracy + Contextual Awareness + Response Quality) / 4

### Run Evaluation

```bash
python scripts/evaluate.py
```

### Evaluation Output

- `evaluation/results.json`: Detailed results in JSON format
- `evaluation/results.csv`: Results in CSV format for analysis

## Report Generation

### Generate PDF Report

```bash
python scripts/generate_report.py
```

### Report Sections

1. **Overview**: Problem statement, objective, RAG approach, source documents
2. **Technical Implementation**: PDF ingestion, chunking, embeddings, FAISS, retrieval, prompts, LLM, RAG chain, memory, source attribution
3. **Evaluation**: Questions, framework, methodology, metrics
4. **Results & Discussion**: Actual results, strengths, limitations, challenges
5. **Conclusion**: Summary, performance, learnings, future improvements

## Testing

### Run Tests

```bash
pytest
```

### Test Coverage

- PDF loading and validation
- Text cleaning
- Document chunking
- Conversation memory (including sliding window behavior)
- Evaluation questions (count, structure, unique IDs)

## Troubleshooting

### Ollama Not Available

**Error**: "Ollama is not available. Please start Ollama."

**Solution**:
```bash
# Start Ollama
ollama serve

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Model Unavailable

**Error**: "Generation model llama3.2:3b is unavailable."

**Solution**:
```bash
# Pull generation model
ollama pull llama3.2:3b

# Pull embedding model
ollama pull nomic-embed-text

# Verify models
ollama list
```

### PDFs Missing

**Error**: "Required PDF 1706.03762.pdf was not found in data/pdfs/."

**Solution**: Download and place PDFs in data/pdfs/:
```bash
# Download from arxiv
wget https://arxiv.org/pdf/1706.03762.pdf -P data/pdfs/
wget https://arxiv.org/pdf/1810.04805.pdf -P data/pdfs/
wget https://arxiv.org/pdf/2005.14165.pdf -P data/pdfs/
wget https://arxiv.org/pdf/1907.11692.pdf -P data/pdfs/
wget https://arxiv.org/pdf/1910.10683.pdf -P data/pdfs/
```

### Vector Store Missing

**Error**: "Vector store not found."

**Solution**: Run ingestion script:
```bash
python scripts/ingest.py
```

### Import Errors

**Error**: ModuleNotFoundError for langchain packages

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

## Ollama Setup

### Install Ollama

Download and install Ollama from: https://ollama.com/download

### Pull Models

```bash
# Pull generation model
ollama pull llama3.2:3b

# Pull embedding model
ollama pull nomic-embed-text

# Verify installation
ollama list
```

Expected output:
```
NAME              ID              SIZE    MODIFIED
llama3.2:3b       ...             ...     ...
nomic-embed-text   ...             ...     ...
```

### Start Ollama

```bash
ollama serve
```

Ollama will be available at http://localhost:11434

## Application Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows**:
```bash
.venv\Scripts\activate
```

**Linux/macOS**:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

Copy `.env.example` to `.env` and customize if needed:
```bash
cp .env.example .env
```

Default configuration:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=5
LOG_LEVEL=INFO
```

### 5. Download PDFs

Manually download the five required PDFs and place them in `data/pdfs/`:
- 1706.03762.pdf
- 1810.04805.pdf
- 2005.14165.pdf
- 1907.11692.pdf
- 1910.10683.pdf

### 6. Run Ingestion

```bash
python scripts/ingest.py
```

This will:
- Load PDFs
- Extract and clean text
- Chunk documents
- Generate embeddings
- Create FAISS vector store
- Persist vector store

### 7. Start Chatbot

```bash
python scripts/chat.py
```

### 8. Run Tests

```bash
pytest
```

### 9. Run Evaluation

```bash
python scripts/evaluate.py
```

### 10. Generate Report

```bash
python scripts/generate_report.py
```

## Project Structure

```
generative-ai-training/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py            # PDF loading with PyMuPDFLoader
│   │   ├── cleaner.py           # Text preprocessing
│   │   └── chunker.py           # Document chunking
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embeddings.py        # Ollama embeddings (nomic-embed-text)
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── faiss_store.py       # FAISS vector store management
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py         # Semantic retrieval
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama.py            # ChatOllama (llama3.2:3b)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompts.py           # RAG prompt templates
│   │   ├── document_formatter.py # Document formatting for context
│   │   └── rag_chain.py         # RAG chain with LCEL
│   └── memory/
│       ├── __init__.py
│       └── conversation_memory.py # Conversation memory (deque maxlen=4)
├── scripts/
│   ├── chat.py                  # CLI chatbot
│   ├── ingest.py               # Ingestion pipeline
│   ├── evaluate.py             # Evaluation script
│   └── generate_report.py       # PDF report generator
├── tests/
│   ├── __init__.py
│   ├── test_loader.py           # PDF loader tests
│   ├── test_cleaner.py          # Text cleaner tests
│   ├── test_chunker.py          # Chunker tests
│   ├── test_memory.py           # Memory tests
│   └── test_evaluation.py       # Evaluation questions tests
├── evaluation/
│   ├── questions.json           # 10 evaluation questions
│   ├── results.json             # Evaluation results (generated)
│   └── results.csv             # Evaluation results (generated)
├── data/
│   ├── pdfs/                    # Place PDFs here (not in git)
│   │   ├── 1706.03762.pdf
│   │   ├── 1810.04805.pdf
│   │   ├── 2005.14165.pdf
│   │   ├── 1907.11692.pdf
│   │   └── 1910.10683.pdf
│   └── vectorstore/             # FAISS index (generated, not in git)
├── reports/
│   └── final_report.pdf         # Generated report (not in git)
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Configuration

### Environment Variables

- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Generation model (default: llama3.2:3b)
- `OLLAMA_EMBEDDING_MODEL`: Embedding model (default: nomic-embed-text)
- `CHUNK_SIZE`: Chunk size in characters (default: 800)
- `CHUNK_OVERLAP`: Chunk overlap in characters (default: 120)
- `TOP_K`: Number of documents to retrieve (default: 5)
- `LOG_LEVEL`: Logging level (default: INFO)

## Summary

**Generation**: Ollama → llama3.2:3b

**Embeddings**: Ollama → nomic-embed-text

**Vector Database**: FAISS

**RAG Framework**: LangChain

**Memory**: Last 4 interaction pairs

**Interface**: CLI

## License

This project is created for educational purposes for the Generative AI Fundamentals assignment.
