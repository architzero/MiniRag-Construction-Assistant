"""
Document Chunker for Mini RAG Project
Sentence-aware, semantically safe chunking
"""
import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    metadata: Dict


class DocumentChunker:
    def __init__(self, max_chars: int = 500, overlap_sentences: int = 2):
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def clean_text(self, text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        raw = re.split(r'(?<=[.!?])\s+|\n+', text)

        sentences = []
        for s in raw:
            s = s.strip()
            if len(s) < 30:
                continue
            sentences.append(s)

        return sentences

    def create_chunks(self, text: str, source: str) -> List[Chunk]:
        text = self.clean_text(text)
        sentences = self.split_sentences(text)

        chunks: List[Chunk] = []
        buffer: List[str] = []
        buffer_len = 0
        chunk_id = 0

        for sentence in sentences:
            # If adding sentence exceeds size, flush buffer
            if buffer_len + len(sentence) > self.max_chars and buffer:
                chunk_text = " ".join(buffer)

                # Avoid duplicate consecutive chunks
                if not chunks or chunks[-1].text != chunk_text:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            source=source,
                            chunk_id=chunk_id,
                            metadata={
                                "char_length": len(chunk_text),
                                "sentences": len(buffer)
                            }
                        )
                    )
                    chunk_id += 1

                # Sentence-based overlap
                buffer = buffer[-self.overlap_sentences:]
                buffer_len = sum(len(s) for s in buffer)

            buffer.append(sentence)
            buffer_len += len(sentence)

        # Flush remaining buffer
        if buffer:
            chunk_text = " ".join(buffer)
            if not chunks or chunks[-1].text != chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=source,
                        chunk_id=chunk_id,
                        metadata={
                            "char_length": len(chunk_text),
                            "sentences": len(buffer)
                        }
                    )
                )

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Chunk]:
        all_chunks = []

        print("\n📄 Chunking documents")
        print(f"   Max chars per chunk: {self.max_chars}")
        print(f"   Overlap (sentences): {self.overlap_sentences}\n")

        for doc in documents:
            doc_chunks = self.create_chunks(doc["text"], doc["filename"])
            all_chunks.extend(doc_chunks)
            print(f"   {doc['filename']} → {len(doc_chunks)} chunks")

        print(f"\n✅ Total chunks created: {len(all_chunks)}")
        return all_chunks


def main():
    from document_processor import DocumentProcessor

    processor = DocumentProcessor(data_dir="data")
    documents = processor.process_all_documents()

    if not documents:
        print("❌ No documents found")
        return

    chunker = DocumentChunker(max_chars=500, overlap_sentences=2)
    chunks = chunker.chunk_documents(documents)

    print("\n🔍 Sample chunks:\n")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"Chunk {i} ({chunk.source})")
        print(chunk.text[:200])
        print("-" * 40)


if __name__ == "__main__":
    main()
