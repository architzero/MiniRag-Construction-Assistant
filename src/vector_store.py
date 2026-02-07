import faiss
import numpy as np
import json
import os

class VectorStore:
    def __init__(self, dimension=384, index_path="index/assignment"):
        self.dimension = dimension
        self.index_path = index_path
        # IndexFlatIP calculates inner product, which equals cosine similarity for normalized vectors
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add(self, embeddings, metadata_list):
        if len(metadata_list) != len(embeddings):
            raise ValueError("Number of embeddings must match number of metadata entries.")
        
        # Ensure float32 for FAISS
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Defensive normalization (optional but robust)
        faiss.normalize_L2(embeddings_np)
        
        self.index.add(embeddings_np)
        self.metadata.extend(metadata_list)

    def search(self, query_vector, k=3):
        # Ensure query is 2D array (1, dimension) and float32
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        
        # Defensive normalization to match the index
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 if no match
                results.append({
                    "text": self.metadata[idx]["text"],
                    "source": self.metadata[idx]["source"],
                    "score": float(distances[0][i])
                })
        return results

    def save(self):
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
            
        faiss.write_index(self.index, os.path.join(self.index_path, "vector_store.index"))
        
        # Save readable JSON metadata (indent=4 is great for debugging)
        with open(os.path.join(self.index_path, "vector_store.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)
        print(f"Index saved to {self.index_path}")

    def load(self):
        index_file = os.path.join(self.index_path, "vector_store.index")
        metadata_file = os.path.join(self.index_path, "vector_store.json")
        
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            print(" No existing index found.")
            return

        self.index = faiss.read_index(index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f" Index loaded with {self.index.ntotal} documents.")