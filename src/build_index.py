"""
Build Vector Index for Mini RAG Project
Stable version with guaranteed paths
"""

import sys
from pathlib import Path

# Ensure src is on Python path
sys.path.append(str(Path(__file__).parent))

from document_processor import DocumentProcessor
from chunker import DocumentChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore


def main(index_mode: str = "assignment"):
    print("=" * 60)
    print(f"🚀 BUILDING VECTOR INDEX | MODE: {index_mode.upper()}")
    print("=" * 60)

    # Base index directory
    base_index_dir = Path("index") / index_mode
    base_index_dir.mkdir(parents=True, exist_ok=True)

    index_path = base_index_dir / "vector_store.index"
    meta_path = base_index_dir / "vector_store.json"

    # 1. Load documents
    print("\n📄 Step 1: Loading documents...")
    processor = DocumentProcessor(data_dir="data")
    documents = processor.process_all_documents()

    if not documents:
        print("❌ No documents found. Exiting.")
        return

    # 2. Chunk documents
    print("\n✂️ Step 2: Chunking documents...")
    chunker = DocumentChunker(max_chars=500, overlap_sentences=2)
    chunks = chunker.chunk_documents(documents)

    if not chunks:
        print("❌ No chunks created. Exiting.")
        return

    # 3. Generate embeddings
    print("\n🔢 Step 3: Generating embeddings...")
    embedder = EmbeddingGenerator()
    embeddings, texts, metadata = embedder.embed_chunks(chunks)

    # 4. Build vector store
    print("\n📦 Step 4: Building vector store...")
    vector_store = VectorStore(embedding_dim=embedder.embedding_dim)
    vector_store.build(embeddings, texts, metadata)

    # 5. Save index
    print("\n💾 Step 5: Saving index to disk...")
    vector_store.save(
        index_path=str(index_path),
        meta_path=str(meta_path)
    )

    print("\n✅ INDEX BUILD COMPLETE")
    print(f"📁 Saved to: {base_index_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main(index_mode="assignment")
