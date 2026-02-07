from pathlib import Path

class RAGPipeline:
    def __init__(
        self,
        embedder,
        index_mode: str = "assignment",
        llm_model: str = "llama3.2:3b",
        top_k: int = 3,
        temperature: float = 0.1
    ):
        self.embedder = embedder
        self.llm_model = llm_model
        self.top_k = top_k
        self.temperature = temperature

        index_dir = Path("index") / index_mode

        self.vector_store = VectorStore(
            embedding_dim=embedder.embedding_dim
        )
        self.vector_store.load(
            index_path=str(index_dir / "vector_store.index"),
            meta_path=str(index_dir / "vector_store.json")
        )

        print(f"🤖 RAG using index mode: {index_mode}")
