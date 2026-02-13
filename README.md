### MiniRAG - Grounded Construction Intelligence Assistant

A production-ready RAG system that answers construction marketplace queries using only internal documentation. No hallucinations, full source traceability.

🔗 **[Live Demo](https://minirag-construction-assistant-5qjcmvpiyucdiiekunzbfd.streamlit.app/)**

### Overview

MiniRAG is a grounded AI assistant built for construction marketplace queries. It pulls information from internal policy, pricing, and spec documents, then generates answers strictly from what it finds.

#### What It Does
1. Semantic search with FAISS + MiniLM embeddings
2. Context-grounded answer generation (no making stuff up)
3. Shows similarity scores for transparency
4. Smart document chunking that respects headers
5. Two modes: pre-loaded docs or upload your own
6. Works with Groq API (cloud) or Ollama (local)

### How It Works

```
User asks a question
     ↓
Convert to embedding (MiniLM)
     ↓
Search vector database (FAISS)
     ↓
Grab top 5 relevant chunks
     ↓
Feed to LLM with strict instructions
     ↓
Get answer + sources + confidence scores
```

#### Why This Architecture?

- **Retrieval**: Find the right info fast
- **Grounding**: Stop the AI from making things up
- **Generation**: Get structured, cited answers
- **Frontend**: See exactly where answers come from

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

### Technical Details

#### Document Chunking
LLMs can't handle entire documents at once, so we break them into smart chunks:

- Respects markdown headers (#, ##)
- Tracks section hierarchy
- ~600 characters per chunk
- 25% overlap so context doesn't get lost
- Each chunk knows where it came from

Example:
```
[doc2.md | Section: Pricing > Premier]
Steel: JSW or Jindal Neo up to ₹74,000/MT
```

This way each chunk makes sense on its own and you can trace it back to the source.

#### Embeddings & Search

**Model**: sentence-transformers/all-MiniLM-L6-v2  
**Why**: Fast, accurate, runs on CPU

**Search**: FAISS with cosine similarity  
**Returns**: Text chunks + source files + confidence scores

FAISS keeps everything local and deterministic - no cloud dependencies for the core search.

#### Grounding (Anti-Hallucination)

The LLM gets strict instructions:
- Only use the provided context
- Don't use general knowledge
- If info is missing, say "I don't have enough information"
- Cite sources when possible

Plus the UI shows you the exact chunks used, so you can verify everything.

### LLM Options

**Groq API** (default for deployment)
- Fast inference with Llama 3.3 70B
- No local setup needed
- Free tier available

**Ollama** (local option)
- Runs Llama 3.2 3B on your machine
- Fully offline
- Good for privacy-sensitive work

Switch between them in the sidebar.

### Try It Out

🚀 **[Live Demo](https://minirag-construction-assistant-5qjcmvpiyucdiiekunzbfd.streamlit.app/)**

### Run Locally

```bash
# Clone repo
git clone https://github.com/architzero/MiniRag-Construction-Assistant.git
cd MiniRag-Construction-Assistant

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run frontend/app.py
```

The index is already built, so you can start using it right away.

### For Assignment Reviewers

If you need strict reproducibility against source PDFs:

1. Store original files in `data/raw/`
2. Record SHA256 checksums in `data/raw/CHECKSUMS.txt`
3. Add a conversion script that generates `data/doc*.md` from raw files
4. Rebuild the index with `python src/build_index.py`

This keeps the RAG pipeline reviewable end-to-end.
