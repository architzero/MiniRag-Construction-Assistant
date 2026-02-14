import os
import glob
from embedder import Embedder
from vector_store import VectorStore

# Configuration
CHUNK_SIZE = 600       # Approx 100-150 words, good for MiniLM context limit
CHUNK_OVERLAP = 150    # ~25% overlap to maintain context across boundaries

def load_documents_from_folder(folder_path):
    # Sort files for deterministic indexing order
    file_paths = sorted(glob.glob(os.path.join(folder_path, "*.md")))
    documents = []
    
    if not file_paths:
        print(f" No Markdown files found in {folder_path}!")
        return []

    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    print(f" Skipping empty file: {path}")
                    continue
                documents.append({"source": os.path.basename(path), "text": content})
        except Exception as e:
            print(f" Error reading {path}: {e}")
            
    return documents

def advanced_chunking(text, source_name):
    """
    Splits text while tracking Markdown Headers (#, ##).
    Prepends context (e.g., "Section: Pricing > Premier") to every chunk.
    """
    if not text:
        return []
        
    lines = text.split('\n')
    chunks = []
    
    current_context = source_name.split('.')[0]
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Update Context based on Headers
        if line.startswith("# "):
            current_context = line.replace("# ", "").strip()
        elif line.startswith("## "):
            main_topic = current_context.split(" > ")[0]
            current_context = f"{main_topic} > {line.replace('## ', '').strip()}"
            
        # Check size limit
        if current_length + len(line) > CHUNK_SIZE:
            chunk_text = " ".join(current_chunk)
            enriched_text = f"[{source_name} | Section: {current_context}]\n{chunk_text}"
            chunks.append(enriched_text)
            
            # Create Overlap
            current_chunk = current_chunk[-3:] 
            current_length = sum(len(l) for l in current_chunk)
            
        current_chunk.append(line)
        current_length += len(line)

    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        enriched_text = f"[{source_name} | Section: {current_context}]\n{chunk_text}"
        chunks.append(enriched_text)
        
    return chunks

def run_indexing_pipeline(input_docs, output_path):
    """
    Reusable function to index ANY list of documents.
    """
    print(f"Indexing {len(input_docs)} documents to {output_path}...")
    
    embedder = Embedder()
    vector_store = VectorStore(index_path=output_path)
    
    all_chunks = []
    all_metadata = []

    for doc in input_docs:
        chunks = advanced_chunking(doc['text'], doc['source'])
        print(f"Created {len(chunks)} chunks from {doc['source']}")
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "source": doc['source'],
                "text": chunk
            })

    if not all_chunks:
        print("No valid chunks to index.")
        return

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embedder.embed(all_chunks)

    vector_store.add(embeddings, all_metadata)
    vector_store.save()
    print(f"Indexing Complete - {len(all_chunks)} chunks saved to {output_path}")

if __name__ == "__main__":
    # Default Assignment Mode
    docs = load_documents_from_folder("data")
    if docs:
        run_indexing_pipeline(docs, "index/assignment")