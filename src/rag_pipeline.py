import os
import ollama
from openai import OpenAI
from embedder import Embedder
from vector_store import VectorStore

class RAGPipeline:
    def __init__(self, index_path="index/assignment", model_name="llama3.2:3b"):
        print(f"Loading RAG Pipeline using model: {model_name}...")
        self.vector_store = VectorStore(index_path=index_path)
        self.vector_store.load()
        self.embedder = Embedder()
        self.model_name = model_name
        
    def load_index(self, index_path):
        print(f"🔄 Loading Index from: {index_path}")
        self.vector_store = VectorStore(index_path=index_path)
        self.vector_store.load()

    def retrieve(self, query, k=5):
        """
        Retrieves top-k chunks. 
        k=5 is maintained to ensure high recall for "Package Pricing" vs "Allowances".
        """
        print(f"🔍 Query: {query}")
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(query_embedding, k=k)
        return results

    def generate_answer(self, query, context_chunks, chat_history=None, model_type="Local (Ollama)"):
        if chat_history is None:
            chat_history = []

        if not context_chunks:
            return "I don't have enough information to answer that."

        # Prepare Context
        context_text = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks])
        
        # Prepare History (Shortened)
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-2:]])

        # YOUR PREFERRED PROMPT
        system_prompt = f"""You are a RAG assistant for Indecimal. Follow these rules STRICTLY:
        1. ONLY use information from the Context below.
        2. Do NOT use your general knowledge.
        3. If the answer is not in Context, say: "I don't have enough information to answer that."
        4. Cite sources when possible (e.g., "According to doc2.md...").
        5. Be concise and factual.

        Context:
        {context_text}
        
        Chat History:
        {history_text}
        """
        
        if model_type == "Local (Ollama)":
            return self._call_ollama(system_prompt, query)
        elif model_type == "OpenRouter API":
            return self._call_openrouter(system_prompt, query)
        else:
            return " Error: Invalid Model Type Selected"

    def _call_ollama(self, system_prompt, query):
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': query}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f" Ollama Error: {str(e)} (Is Ollama running?)"

    def _call_openrouter(self, system_prompt, query):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return " Error: OpenRouter API Key is missing. Please enter it in the Sidebar."

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        try:
            # Using Mistral 7B as a safe default
            response = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f" OpenRouter Error: {str(e)}"

    def run(self, query, chat_history=None, model_type="Local (Ollama)"):
        if chat_history is None:
            chat_history = []
            
        retrieved_chunks = self.retrieve(query)
        answer = self.generate_answer(query, retrieved_chunks, chat_history, model_type)
        
        return {
            "answer": answer,
            "sources": retrieved_chunks
        }