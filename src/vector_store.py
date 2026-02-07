"""
Vector Store for Mini RAG Project
FAISS-based vector store using cosine similarity
"""

import faiss
import numpy as np
import json
from typing import List, Dict


class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        self.texts: List[str] = []
        self.metadata: List[Dict] = []

    def build(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict]):
        print("\n🏗️ Building FAISS index (cosine similarity)")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype("float32"))

        self.texts = texts
        self.metadata = metadata

        print(f"   Indexed vectors: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3, min_similarity: float = 0.3):
        if self.index is None:
            raise RuntimeError("Vector index not loaded")

        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)

        sims, idxs = self.index.search(query_embedding, top_k)

        results = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx == -1 or sim < min_similarity:
                continue

            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(sim)
            })

        return results

    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_dim": self.embedding_dim,
                    "texts": self.texts,
                    "metadata": self.metadata
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"💾 Vector store saved to {index_path}")

    def load(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.embedding_dim = data["embedding_dim"]
            self.texts = data["texts"]
            self.metadata = data["metadata"]

        print(f"📂 Vector store loaded from {index_path}")
