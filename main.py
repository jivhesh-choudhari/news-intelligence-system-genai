import os
import nltk
from sentence_transformers import SentenceTransformer

from ingestion.fetcher import start_background_polling
from ingestion.simulator import start_background_streaming
from rag.generator import Generator
from rag.evaluate import RAGEvaluator

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

RAG_DIR = os.path.join(os.path.dirname(__file__), "rag")


def main():
    print("Starting background services...")
    start_background_polling(interval=600)
    start_background_streaming(mode="sessions")
    print("News polling and session streaming started.\n")

    print("Loading models...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    generator = Generator(
        index_path=os.path.join(RAG_DIR, "faiss.index"),
        bm25_path=os.path.join(RAG_DIR, "bm25.db"),
        metadata_path=os.path.join(RAG_DIR, "metadata.db"),
        embedding_model=embedding_model,
    )
    evaluator = RAGEvaluator(embedding_model=embedding_model)
    print("Ready.\n")

    while True:
        try:
            question = input("prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            break

        result = generator.generate(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Confidence: {result['top_score']:.3f}")

        scores = evaluator.evaluate(question, result["answer"], result["chunks"])
        if scores["faithfulness"] is not None:
            print(f"Faithfulness: {scores['faithfulness']:.2f}  Relevance: {scores['relevance']:.2f}")

        print()


if __name__ == "__main__":
    main()

