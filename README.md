# NLPv3 - RAG-Based College FAQ Chatbot

A CLI-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer student queries about college admissions, fees, exams, library, hostel, and campus facilities. It retrieves relevant information from a curated knowledge base and generates answers using an LLM via Ollama.

## Project Structure

```
NLPv3/
├── data/
│   ├── raw/                     # Source documents (DOCX, PDF, TXT, HTML, MD)
│   └── output/
│       └── chunks.json          # Processed text chunks
├── src/
│   ├── text_processor.py        # Document extraction, cleaning, chunking
│   └── rag_engine.py            # FAISS indexing, retrieval, LLM generation
├── prompts/
│   ├── system_prompt.txt        # System prompt for the LLM
│   └── few_shot_examples.json   # Few-shot examples
├── models/
│   └── faiss_index/             # Generated FAISS vector index
├── requirements.txt
└── README.md
```

## Pipeline

### Phase 1: Document Processing (`text_processor.py`)
- Reads documents from `data/raw/` (supports PDF, DOCX, TXT, HTML, MD)
- Extracts text using format-specific extractors
- Cleans and normalizes text
- Splits into 400-character chunks with 50-character overlap using LangChain
- Outputs `data/output/chunks.json`

### Phase 2: FAISS Indexing & RAG (`rag_engine.py`)
- Generates embeddings using `all-MiniLM-L6-v2` (sentence-transformers)
- Builds a FAISS vector index (Inner Product for cosine similarity)
- Retrieves top-k relevant chunks for a given query
- Sends query + context to Ollama (`llama3.2`) with system prompt and few-shot examples
- Returns answer with source references

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
Install Ollama from https://ollama.com, then pull the model:
```bash
ollama pull llama3.2
```

### 3. Add documents
Place your files (PDF, DOCX, TXT, HTML, MD) in `data/raw/`.

## Usage

### Step 1: Process documents into chunks
```bash
python src/text_processor.py
```
Generates `data/output/chunks.json`.

### Step 2: Build the FAISS index
```bash
python src/rag_engine.py build
```
Generates `models/faiss_index/index.faiss` and `chunks.pkl`.

### Step 3: Ask a question
```bash
python src/rag_engine.py query "What are the admission requirements?"
```
