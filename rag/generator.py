from huggingface_hub import InferenceClient
from dotenv import load_dotenv, find_dotenv
from retrieval import HybridRetriever
from guardrails import Guardrails
import os


load_dotenv(find_dotenv())

SYSTEM_PROMPT = (
    "You are a news assistant. Answer questions using only the provided context. "
    "If the context does not contain enough information to answer, say exactly: I don't know."
)

class Generator:
    def __init__(
        self,
        index_path="faiss.index",
        bm25_path="bm25.db",
        metadata_path="metadata.db",
        model="Qwen/Qwen2.5-7B-Instruct"
    ):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError("HF_TOKEN not set. Add HF_TOKEN=<your_token> to your .env file.")

        self.retriever = HybridRetriever(index_path, bm25_path, metadata_path)
        self.client = InferenceClient(model=model, token=token)
        self.guardrails = Guardrails()

    def generate(self, question: str) -> dict:
        answer = ""
        if self.guardrails.check_injection(question):
            return {"answer": "I can only answer questions about news topics.", "chunks": [], "top_score": 0.0}
        
        docs = self.retriever.retrieve(question, top_k=5)
        if not docs:
            return {"answer": "I don't know.", "chunks": [], "top_score": 0.0}

        top_score = docs[0][1]
        chunks = [content for _, _, content in docs]
        if self.guardrails.check_abstention(top_score):
            return {"answer": "I don't have enough information to answer this.", "chunks": [], "top_score": 0.0}
        else:
            context = "\n\n".join([f"[{i+1}] {content}" for i, (_, _, content) in enumerate(docs)])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
            response = self.client.chat_completion(messages, max_tokens=512, temperature=0.1)
            answer = response.choices[0].message.content.strip()

        return {"answer": answer, "chunks": chunks, "top_score": top_score}


if __name__ == "__main__":
    generator = Generator()
    print("News Intelligence System")
    print("Type 'exit' to quit\n")

    while True:
        question = input("prompt: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        result = generator.generate(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Top score: {result['top_score']:.4f}\n")
        for i, chunk in enumerate(result['chunks']):
            print(f"[{i+1}] {chunk}\n")