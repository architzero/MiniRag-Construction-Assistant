import time
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

def run_evaluation():
    print("🚀 Starting Automated Evaluation...")
    rag = RAGPipeline(index_path="index/assignment")
    
    results = []
    total_start = time.time()

    for q in TEST_QUESTIONS:
        print(f"🔹 Testing: {q}")
        
        start_time = time.time()
        response = rag.run(q)
        latency = time.time() - start_time
        
        top_score = response["sources"][0]["score"] if response["sources"] else 0
        
        results.append({
            "Question": q,
            "Answer_Length": len(response["answer"]),
            "Latency_Seconds": round(latency, 4),
            "Top_Confidence": round(top_score, 4),
            "Sources_Retrieved": len(response["sources"])
        })

    total_time = time.time() - total_start
    df = pd.DataFrame(results)
    df.to_csv("evaluation_report.csv", index=False)
    print("\n✅ Evaluation Complete!")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Latency: {df['Latency_Seconds'].mean():.2f}s")
    print(f"Average Confidence: {df['Top_Confidence'].mean():.2f}")
    print("📄 Report saved to 'evaluation_report.csv'")

if __name__ == "__main__":
    run_evaluation()