### MiniRAG - Grounded Construction Intelligence Assistant

A production style Retrieval-Augmented Generation (RAG) system designed to answer construction marketplace queries using strictly internal documentation, with full source traceability and zero hallucination.

### Overview

MiniRAG is a grounded AI assistant built for a construction marketplace use case.
It retrieves information from internal policy, pricing and specification documents and generates answers from retrieved context.

#### Key Features
1. Semantic Retrieval using FAISS + MiniLM embeddings
2. Strict Context Grounded Answer Generation
3. Similarity Score Transparency
4. Header-Aware Intelligent Chunking
5. Dual Mode Indexing (Assignment Docs + Custom Uploads)
6. Flexible LLM Backend (Local via Ollama or API-based)

### System Architecture

```
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
```

#### Architectural Philosophy

Retrieval Layer: Precision & Relevance
Grounding Layer: Hallucination Control
Generation Layer: Structured Answering
Frontend Layer: Transparency & Usability

#### Project Structure

```
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
```

### Technical Deep Dive

#### Intelligent Document Chunking
Large language models cannot process entire documents reliably due to context window limits.
To address this, documents are split into semantic chunks using a context-aware strategy.

Chunking strategy:

1. Markdown header-aware parsing (#, ##)
2. Hierarchical section tracking
3. ~600 character chunk size
4. ~25% overlap for semantic continuity
4. Metadata injected into each chunk

Example chunk format:

```
[doc2.md | Section: Pricing > Premier]
Steel: JSW or Jindal Neo up to ₹74,000/MT
```

This ensures that each chunk:

1. Is meaningful in isolation
2. Retains section identity
3. Avoids cross-section confusion during retrieval

#### Embeddings

```
Model: sentence-transformers/all-MiniLM-L6-v2
Vector Dimension: 384
Normalization: L2-normalized
Similarity Metric: Cosine similarity
```

MiniLM was chosen because it:
1. Performs well on short semantic text
2. Runs efficiently on CPU
3. Is widely accepted for retrieval tasks

#### Vector Search (FAISS)

Uses FAISS IndexFlatIP

Inner Product + normalized vectors = cosine similarity

Top-K retrieval (default: 5)

Returns:

1. Chunk text
2. Source document
3. Similarity score

FAISS is used locally to keep the system:

1. Lightweight
2. Deterministic
3. Easy to evaluate

#### Strict Grounding Enforcement

The core safety mechanism is prompt-level grounding.

System rules enforced during generation:

1. Use only the provided context
2. Do not use external or general knowledge
3. If information is missing, explicitly respond:
   “I don’t have enough information to answer that.”

Additional safeguards:

1. Context-only prompt injection
2. No open-ended generation
3. Source visibility in UI
4. Similarity score transparency

### LLM Backend Options

The system supports two execution modes:

#### Local (Offline)

LLaMA 3.2 (3B) via Ollama

* Fully offline inference
* Suitable for privacy-sensitive workflows

#### API-Based

OpenRouter supported models (e.g. Mistral-7B)

* No GPU required
* Easy experimentation

The backend can be switched from the Streamlit sidebar.

### Frontend

The frontend allows clear visibility of:

* Answers
* Sources
* Similarity scores
* Index mode

### Getting Started

* Prerequisites

1. Python 3.8+
2. Optional: Ollama (for local LLM mode)

```
# Clone Repositiory
git clone https://github.com/architzero/MiniRag-Construction-Assistant.git
cd MiniRag-Construction-Assistant

# Install Dependencies
pip install -r requirements.txt

# Build Assignment Index
python src/build_index.py

# Run Backend Test (Optional)
python test_rag.py

# Launch Frontend
streamlit run frontend/app.py
```
## Document provenance checklist (for assignment reviewers)

If you need strict reproducibility against source PDFs:

1. Store original files in `data/raw/`.
2. Record SHA256 checksums in `data/raw/CHECKSUMS.txt`.
3. Add a conversion script that generates `data/doc*.md` from raw files.
4. Rebuild the index with `python src/build_index.py`.

This keeps the RAG pipeline reviewable end-to-end.