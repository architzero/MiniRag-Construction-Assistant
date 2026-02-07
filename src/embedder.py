import torch
from sentence_transformers import SentenceTransformer
from typing import Union, List
import numpy as np

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Check for GPU/MPS (Mac) or default to CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading embedding model: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embeds a list of strings or a single string.
        Returns a numpy array of float32 embeddings.
        """
        # If a single string is passed, wrap it in a list
        if isinstance(texts, str):
            texts = [texts]
        
        # normalize_embeddings=True ensures cosine similarity via dot product
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)