"""
Embeddings Generator for Mini RAG Project
Creates vector embeddings using sentence-transformers
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
from pathlib import Path

class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model name
                      Default: all-MiniLM-L6-v2 (384 dimensions, fast & efficient)
        """
        self.model_name = model_name
        print(f"🤖 Loading embedding model: {model_name}")
        
        # Create cache directory
        cache_dir = Path("models/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded! Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embeddings
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_chunks(self, chunks: List) -> tuple:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Tuple of (embeddings array, chunk texts, chunk metadata)
        """
        print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
        
        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts, show_progress=True)
        
        # Prepare metadata
        metadata = [
            {
                'source': chunk.source,
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dimension: {self.embedding_dim}")
        
        return embeddings, texts, metadata
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to disk"""
        np.save(filepath, embeddings)
        print(f"💾 Embeddings saved to: {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from disk"""
        embeddings = np.load(filepath)
        print(f"📂 Embeddings loaded from: {filepath}")
        return embeddings

def main():
    """Test embedding generation"""
    from document_processor import DocumentProcessor
    from chunker import DocumentChunker
    
    # Load and chunk documents
    print("Step 1: Loading documents...")
    processor = DocumentProcessor()
    documents = processor.process_all_documents()
    
    if not documents:
        print("No documents found!")
        return
    
    print("\nStep 2: Chunking documents...")
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)
    
    if not chunks:
        print("No chunks created!")
        return
    
    # Generate embeddings
    print("\nStep 3: Generating embeddings...")
    embedder = EmbeddingGenerator()
    embeddings, texts, metadata = embedder.embed_chunks(chunks)
    
    # Save embeddings
    embedder.save_embeddings(embeddings, "embeddings.npy")
    
    print(f"\n{'='*50}")
    print("✅ Embedding generation complete!")
    print(f"{'='*50}")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Memory size: {embeddings.nbytes / 1024:.2f} KB")

if __name__ == "__main__":
    main()