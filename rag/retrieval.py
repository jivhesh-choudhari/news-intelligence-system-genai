# implement a retriever that uses FAISS Dense Seacrch combined with BM25 sparse search to retrieve relevant documents from a large corpus. The retriever should take a query as input and return a list of relevant documents based on both dense and sparse search results.
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
try:
    from .fusion import fusion
except ImportError:
    from fusion import fusion
import numpy as np
import sentence_transformers
import faiss
import sqlite3

#source of metadata metadata.db
#source of index faiss.index
class HybridRetriever:
    def __init__(self, index_path, bm25_path, metadata_path, model_name='all-MiniLM-L6-v2'):
        self.index = faiss.read_index(index_path)
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.meta_conn = sqlite3.connect(metadata_path)
        self.bm25_conn = sqlite3.connect(bm25_path)
        self.bm25_cursor = self.bm25_conn.cursor()
        self.meta_cursor = self.meta_conn.cursor()

    def retrieve(self, query, top_k=5):
        
        query_embedding = self.model.encode([query]).astype("float32")
        D, I = self.index.search(query_embedding.reshape(1, -1), top_k)
        FAISS_results = np.array(list(zip(I[0], D[0]))[:top_k])
        
        self.bm25_cursor.execute("""
            SELECT chunk_id, tokens
            FROM bm25_docs
        """)
        rows = self.bm25_cursor.fetchall()
        chunk_ids = []
        tokenized_corpus = []
        for chunk_id, tokens in rows:
            chunk_ids.append(chunk_id)
            tokenized_corpus.append(
                tokens.split()
            )
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = word_tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_results = [(chunk_ids[i], bm25_scores[i]) for i in range(len(chunk_ids))]
        bm25_results_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)[:top_k]
        BM25_results = np.array(bm25_results_sorted)
        scores = fusion(BM25_results, FAISS_results)
        
        #Retrieve the actual content of the documents based on the combined scores
        doc_ids = [doc_id for doc_id, score in scores]
        placeholders = ', '.join(['?'] * len(doc_ids))
        self.meta_cursor.execute(f"""
            SELECT id, content
            FROM chunks
            WHERE id IN ({placeholders})
        """, doc_ids)
        rows = self.meta_cursor.fetchall()
        docs = {chunk_id: content for chunk_id, content in rows}
        return docs