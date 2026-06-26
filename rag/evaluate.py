import re
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ABSTENTION_RESPONSES = {
    "i don't know.",
    "i don't have enough information to answer this.",
    "i can only answer questions about news topics."
}

class RAGEvaluator:
    def __init__(self, embedding_model: SentenceTransformer = None):
        self.nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1
        )
        self.embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')

    def _split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def evaluate_faithfulness(self, answer, chunks):
        sentences = self._split_sentences(answer)
        if not sentences or not chunks:
            return 0.0

        grounded = 0
        for sentence in sentences:
            for chunk in chunks:
                result = self.nli({"text": chunk, "text_pair": sentence}, top_k=None)
                entailment_score = next(
                    (r["score"] for r in result if r["label"].lower() in ("entailment", "label_1")),
                    0.0
                )
                if entailment_score > 0.5:
                    grounded += 1
                    break

        return grounded / len(sentences)

    def evaluate_relevance(self, question, chunks):
        if not chunks:
            return 0.0
        question_emb = self.embedding_model.encode([question], normalize_embeddings=True)
        chunk_embs = self.embedding_model.encode(chunks, normalize_embeddings=True)
        similarities = cosine_similarity(question_emb, chunk_embs)[0]
        return float(np.mean(similarities))

    def evaluate(self, question, answer, chunks):
        if answer.lower().strip() in ABSTENTION_RESPONSES:
            return {"faithfulness": None, "relevance": None}

        return {
            "faithfulness": self.evaluate_faithfulness(answer, chunks),
            "relevance": self.evaluate_relevance(question, chunks)
        }


if __name__ == "__main__":
    evaluator = RAGEvaluator()

    question = "What did the RBI announce?"
    answer = "The Reserve Bank of India announced a new liquidity framework to improve credit for small businesses."
    chunks = [
        "The Reserve Bank of India (RBI) announced a new liquidity framework today. Analysts say the move could improve credit availability for small businesses.",
        "Markets reacted positively to the announcement. The NIFTY 50 gained 1.2% during afternoon trading."
    ]

    scores = evaluator.evaluate(question, answer, chunks)
    print(f"Faithfulness : {scores['faithfulness']:.2f}")
    print(f"Relevance    : {scores['relevance']:.2f}")
