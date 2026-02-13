## Evaluation & Performance Analysis

This folder contains a lightweight, practical evaluation script for the MiniRAG pipeline.

### What is measured
1. **Latency** per question.
2. **Top retrieval confidence** (highest FAISS similarity score among retrieved chunks).
3. **Sources retrieved** count.
4. **Grounded overlap**: fraction of non-trivial answer words that also appear in retrieved context.
5. **Fallback behavior**: whether the answer used the explicit "I don't have enough information" response.

### Why this is useful
- It gives a quick health check for retrieval and response behavior.
- It highlights potential hallucination risk when overlap is consistently low.

### Caveat
Grounded overlap is a lexical heuristic, not a formal faithfulness metric. Use it for trend monitoring, not as a final correctness guarantee.