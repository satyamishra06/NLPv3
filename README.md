# NLPv3 - RAG-Based College FAQ Chatbot

Students routinely face delays in getting answers to common queries related to admissions, fee structures, examination schedules, and campus facilities. Traditional rule-based FAQ systems are brittle — they fail whenever a question is paraphrased or worded differently from the exact patterns they were programmed to match.

This project proposes a chatbot that combines a Large Language Model (LLM) with **Retrieval-Augmented Generation (RAG)**. Instead of relying purely on the model's internal knowledge, the system retrieves the most relevant information from a curated, college-specific knowledge base and uses it to generate accurate, conversational answers. **Prompt Engineering** is used to constrain the LLM to answer strictly from the retrieved context, reducing hallucinations and keeping responses on-topic and verifiable.

---

## Problem Statement

A large proportion of frequently asked student queries are currently handled through repetitive manual support (front-office staff, email, phone), causing response delays and inconsistent answers across staff members. The challenge is to design a system that gives accurate, context-grounded answers without the cost, time, and data requirements of training a language model from scratch.

---

## Objectives

1. Build a question-answering chatbot capable of answering student queries about college and support-related topics.
2. Implement a RAG pipeline that retrieves relevant document chunks from a knowledge base and uses an LLM to generate grounded answers.
3. Apply prompt engineering techniques — system prompt, few-shot examples, answer-from-context-only constraint — to improve answer quality and reliability.
4. Evaluate the relevance and correctness of chatbot answers, and compare performance with and without RAG.

---

## Scope

- **Domain:** Student support topics only — admissions, fees, examinations, library, and hostel.
- **Knowledge base:** 5–10 PDF or plain-text documents, such as department FAQs and official notices.
- **Interaction mode:** Text-only input and output (no voice interface, no multi-language support).

---

## Project Structure

```
NLPv3/
│
├── data/                        # All data-related files
│   ├── raw/                     # PLACE: Original PDF documents here
│   │   ├── admissions_faq.pdf
│   │   ├── fee_structure.pdf
│   │   ├── exam_schedule.pdf
│   │   ├── library_info.pdf
│   │   └── hostel_rules.pdf
│   │
│   └── output/                  # AUTO-GENERATED: Processed chunks
│       └── chunks.json          # Output from text_processor.py
│
├── src/                         # All source code
│   ├── __init__.py
│   ├── text_processor.py        # Phase 1: PDF extraction, cleaning, chunking
│   ├── rag_engine.py            # Phase 2: FAISS indexing, retrieval, LLM generation
│   └── app.py                   # Phase 4: Streamlit chatbot UI
│
├── notebooks/                   # Jupyter notebooks
│   ├── test_processing.ipynb    # Test Phase 1 pipeline
│   ├── evaluation.ipynb         # Phase 4: RAG vs No-RAG comparison
│   └── playground.ipynb         # Experimental/testing notebook
│
├── testing/                     # Test files and test cases
│   └── test_queries.json        # Sample student queries for evaluation
│
├── models/                      # PLACE: Saved FAISS index and embeddings
│   ├── faiss_index/             # FAISS vector store (generated in Phase 2)
│   └── embeddings_cache/        # Cached embeddings (optional)
│
├── prompts/                     # PLACE: Prompt templates
│   ├── system_prompt.txt        # System prompt for LLM
│   └── few_shot_examples.json   # Few-shot examples for prompt engineering
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Where to Keep What

| What | Where | Who |
|------|-------|-----|
| Original college PDFs | `data/raw/` | Member 1 |
| Cleaned chunked data | `data/output/chunks.json` | Auto-generated |
| Text processing code | `src/text_processor.py` | Member 1 |
| RAG pipeline code | `src/rag_engine.py` | Member 2 |
| Streamlit chatbot UI | `src/app.py` | Member 3 |
| FAISS index files | `models/faiss_index/` | Member 2 |
| System prompt & few-shot | `prompts/` | Member 2 |
| Evaluation notebook | `notebooks/evaluation.ipynb` | Member 3 |
| Test queries | `testing/test_queries.json` | Member 3 |
| Experiment notebooks | `notebooks/playground.ipynb` | Anyone |

---

## Phases

### Phase 1: Document Collection & Processing
- Collect 5-10 PDFs (admissions, fees, exams, library, hostel)
- Extract text using `pdfplumber`
- Clean and normalize text
- Split into 300-500 character chunks with overlap using LangChain
- **Output:** `data/output/chunks.json`

### Phase 2: FAISS Indexing & RAG Pipeline
- Generate embeddings using sentence-transformers or OpenAI embeddings
- Build FAISS vector index from chunked data
- Implement retrieval function (top-k search)
- Connect LLM (OpenAI/local) to generate answers from retrieved context
- Design prompts with system constraint and few-shot examples
- **Output:** `src/rag_engine.py` with `retrieve()` and `generate_answer()`

### Phase 3: Prompt Engineering
- System prompt: "Answer only from the given context; if not found, say 'I don't know'"
- Few-shot examples for common query types
- Answer-from-context-only constraint
- **Output:** `prompts/system_prompt.txt`, `prompts/few_shot_examples.json`

### Phase 4: UI & Evaluation
- Build Streamlit chatbot UI with message history
- Connect UI to RAG backend
- Create evaluation notebook with test queries
- Compare RAG vs No-RAG performance
- Measure accuracy, relevance, "I don't know" rate
- **Output:** `src/app.py`, `notebooks/evaluation.ipynb`

---

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd NLPv3
```

### 2. Create virtual environment    ''' not mandatory'''
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your documents
Place all PDF files in `data/raw/` folder.

### 5. Run Phase 1 - Document Processing
```bash
python src/text_processor.py
```
This will generate `data/output/chunks.json`.

---

## Dependencies

```
pdfplumber
langchain
langchain-text-splitters
sentence-transformers
faiss-cpu
openai
streamlit
```

---

## Team Members

| Member | Role | Responsibility |
|--------|------|----------------|
| Member 1 | Data Engineer | Document collection, cleaning, chunking |
| Member 2 | ML Engineer | RAG pipeline, embeddings, FAISS, LLM integration |
| Member 3 | Frontend + Evaluation | Streamlit UI, testing, evaluation notebook |

---

## License

This project is for educational purposes.
