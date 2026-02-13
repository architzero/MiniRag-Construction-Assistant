import time
import re
import pandas as pd
from rag_pipeline import RAGPipeline

TEST_QUESTIONS = [
    "What is the price of the Premier package?",
    "What is the zero-tolerance policy?",
    "How does the escrow payment model work?",
    "Compare steel brands in Essential vs Pinnacle.",
    "Does Indecimal provide home financing?",
    "What is the structural warranty period?",
    "Who handles the project monitoring?",
    "What happens in the 'Design & Finalisation' stage?"
]


def _normalize_words(text: str):
    return set(re.findall(r"[a-zA-Z0-9']+", text.lower()))


def _grounded_overlap(answer: str, sources):
    """Simple lexical proxy: how much answer vocabulary appears in retrieved context."""
    if not answer.strip() or not sources:
        return 0.0

    context = " ".join([s["text"] for s in sources])
    answer_words = _normalize_words(answer)
    context_words = _normalize_words(context)

    stop_words = {
        "the", "a", "an", "is", "are", "to", "of", "in", "and", "for", "on", "with",
        "that", "this", "it", "as", "be", "or", "by", "from", "at", "if", "not", "i"
    }

    answer_content_words = {w for w in answer_words if w not in stop_words and len(w) > 2}
    if not answer_content_words:
        return 1.0

    supported = len(answer_content_words.intersection(context_words))
    return round(supported / len(answer_content_words), 4)


def run_evaluation():
    print("Starting Automated Evaluation...")
    rag = RAGPipeline(index_path="index/assignment")
    
    results = []
    total_start = time.time()

    for question in TEST_QUESTIONS:
        print(f"Testing: {question}")
        
        start_time = time.time()
        response = rag.run(question)
        latency = time.time() - start_time
        
        top_score = response["sources"][0]["score"] if response["sources"] else 0
        grounded_overlap = _grounded_overlap(response["answer"], response["sources"])
        fallback_used = "i don't have enough information" in response["answer"].lower()

        results.append({
            "Question": question,
            "Answer_Length": len(response["answer"]),
            "Latency_Seconds": round(latency, 4),
            "Top_Confidence": round(top_score, 4),
            "Sources_Retrieved": len(response["sources"]),
            "Grounded_Overlap": grounded_overlap,
            "Fallback_Answer": fallback_used,
        })

    total_time = time.time() - total_start
    df = pd.DataFrame(results)
    df.to_csv("evaluation_report.csv", index=False)
    print("\n Evaluation Complete!")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Latency: {df['Latency_Seconds'].mean():.2f}s")
    print(f"Average Confidence: {df['Top_Confidence'].mean():.2f}")
    print(f"Average Grounded Overlap: {df['Grounded_Overlap'].mean():.2f}")
    print("Report saved to 'evaluation_report.csv'")


if __name__ == "__main__":
     run_evaluation()