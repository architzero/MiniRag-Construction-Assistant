### MiniRAG — Grounded Construction Intelligence Assistant

A production-style Retrieval-Augmented Generation (RAG) system designed to answer construction marketplace queries using strictly internal documentation, with full source traceability and zero hallucination.

### 🚀 Overview

MiniRAG is a grounded AI assistant built for a construction marketplace use case.
It retrieves information from internal policy, pricing, and specification documents and generates answers exclusively from retrieved context.

The system is built around one guiding principle:

Accuracy over creativity.

❌ No internet knowledge

❌ No hallucinated facts

✅ Full source transparency

✅ Deterministic, evaluable behavior

✨ Key Features

🔎 Semantic Retrieval using FAISS + MiniLM embeddings

🧠 Strict Context-Grounded Answer Generation

📊 Similarity Score Transparency

📁 Header-Aware Intelligent Chunking

🔄 Dual Mode Indexing (Assignment Docs + Custom Uploads)

🤖 Flexible LLM Backend (Local via Ollama or API-based)

🖥️ Streamlit-based Interactive Interface for Easy Evaluation

### System Architecture

User Query
   ↓
Query Embedding (MiniLM)
   ↓
FAISS Vector Search (Cosine Similarity)
   ↓
Top-K Relevant Chunks
   ↓
Strict Context Injection
   ↓
LLM Generation
   ↓
Final Answer + Sources + Similarity Scores

Architectural Philosophy

The system follows clear separation of concerns:

Retrieval Layer → Precision & relevance

Grounding Layer → Hallucination control

Generation Layer → Structured answering

Frontend Layer → Transparency & usability

### 📂 Project Structure
.
├── frontend/
│   └── app.py                # Streamlit UI
│
├── src/
│   ├── embedder.py           # MiniLM embedding logic
│   ├── vector_store.py       # FAISS index management
│   ├── build_index.py        # Chunking + indexing pipeline
│   └── rag_pipeline.py       # Retrieval + generation orchestration
│
├── data/                     # Assignment documents
│   ├── doc1.md
│   ├── doc2.md
│   └── doc3.md
│
├── index/                    # Generated FAISS indexes
│   ├── assignment/           # Fixed index for evaluation
│   └── custom/               # Optional user-uploaded documents
│
├── test_rag.py               # Backend-only evaluation script
├── requirements.txt
└── README.md

### Technical Deep Dive
 ## 1️⃣ Intelligent Document Chunking

Large language models cannot process entire documents reliably due to context window limits.
To address this, documents are split into semantic chunks using a context-aware strategy.

Chunking strategy:

Markdown header-aware parsing (#, ##)

Hierarchical section tracking

~600 character chunk size

~25% overlap for semantic continuity

Metadata injected into each chunk

Example chunk format:

[doc2.md | Section: Pricing > Premier]
Steel: JSW or Jindal Neo up to ₹74,000/MT


This ensures that each chunk:

Is meaningful in isolation

Retains section identity

Avoids cross-section confusion during retrieval

## 2️⃣ Embeddings

Model: sentence-transformers/all-MiniLM-L6-v2

Vector Dimension: 384

Normalization: L2-normalized

Similarity Metric: Cosine similarity

MiniLM was chosen because it:

Performs well on short semantic text

Runs efficiently on CPU

Is widely accepted for retrieval tasks

## 3️⃣ Vector Search (FAISS)

Uses FAISS IndexFlatIP

Inner Product + normalized vectors = cosine similarity

Top-K retrieval (default: 5)

Returns:

Chunk text

Source document

Similarity score

FAISS is used locally to keep the system:

Lightweight

Deterministic

Easy to evaluate

## 4️⃣ Strict Grounding Enforcement

The core safety mechanism is prompt-level grounding.

System rules enforced during generation:

Use only the provided context

Do not use external or general knowledge

If information is missing, explicitly respond:

“I don’t have enough information to answer that.”

Additional safeguards:

Context-only prompt injection

No open-ended generation

Source visibility in UI

Similarity score transparency

### LLM Backend Options

The system supports two execution modes:

## Local (Offline)

LLaMA 3.2 (3B) via Ollama

Fully offline inference

Suitable for privacy-sensitive workflows

## API-Based

OpenRouter-supported models (e.g., Mistral-7B)

No GPU required

Easy experimentation

The backend can be switched from the Streamlit sidebar.

### Frontend (Why it Exists)

The frontend is built using Streamlit.

Purpose of the frontend:

Easy access for evaluators

No need to run backend scripts manually

Clear visibility of:

Answers

Sources

Similarity scores

Index mode

The frontend is not the focus of the assignment, but a usability layer to make evaluation straightforward.

### Evaluation Strategy

This project is evaluated qualitatively, which is standard for RAG systems.

Evaluation approach:

Build a fixed document index

Ask factual questions

Verify answers against retrieved sources

Confirm correct refusal when information is missing

Example queries tested:

“What cement is used in the Premier package?”

“How does the company ensure quality assurance?”

“What payment safeguards exist for customers?”

Expected behavior:

Accurate retrieval

Grounded answers

No hallucination

Transparent sources

### Getting Started

Prerequisites

Python 3.8+

Optional: Ollama (for local LLM mode)

1️⃣ Clone Repository
git clone https://github.com/architzero/MiniRag-Construction-Assistant.git
cd MiniRag-Construction-Assistant

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Build Assignment Index
python src/build_index.py

4️⃣ Run Backend Test (Optional)
python test_rag.py

5️⃣ Launch Frontend
streamlit run frontend/app.py


### Design Decisions Summary

FAISS chosen over managed vector DBs for simplicity

MiniLM chosen for CPU-friendly semantic search

Strict grounding enforced to eliminate hallucination

Frontend designed for transparency, not aesthetics


### About This Project

This project was built as a technical assignment to demonstrate:

Practical understanding of RAG systems

Semantic retrieval design

Hallucination mitigation

Clean, modular backend engineering

Transparent AI system behavior

As a fresher engineer, the focus was not just to “make it work”, but to design it in a way that reflects production-oriented thinking.

### License

Developed for educational and evaluation purposes.