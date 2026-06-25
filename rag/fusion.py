from typing import List, Tuple
import numpy as np


def fusion(BM25_results:np.ndarray, FAISS_results:np.ndarray)->List[Tuple[int, float]]:
    doc_scores = []
    if len(BM25_results) == 0 or len(FAISS_results) == 0:
        raise ValueError("BM25_results and FAISS_results must not be empty.")
    if BM25_results.shape[1] != 2 or FAISS_results.shape[1] != 2:
        raise ValueError("BM25_results and FAISS_results must have two columns: [id, score].")

    
    BM25_results[:, 1] = (BM25_results[:, 1] - BM25_results[:, 1].min()) / (BM25_results[:, 1].max() - BM25_results[:, 1].min() + 1e-8)  # Normalizing BM25 scores to [0, 1]
    FAISS_results[:, 1] = (FAISS_results[:, 1].max() - FAISS_results[:, 1]) / (FAISS_results[:, 1].max() - FAISS_results[:, 1].min() + 1e-8)  # Normalizing FAISS scores to [0, 1] and inverting
    
    variance_faiss = np.var(FAISS_results[:, 1])
    variance_bm25 = np.var(BM25_results[:, 1])
    alpha = variance_faiss / (variance_faiss + variance_bm25 + 1e-8)  # Weight for FAISS scores
    
    all_doc_ids = set(BM25_results[:, 0]).union(set(FAISS_results[:, 0]))
    for doc_id in all_doc_ids:
        bm25_score = next((score for id, score in BM25_results if id == doc_id), 0)
        faiss_score = next((score for id, score in FAISS_results if id == doc_id), 0)
        combined_score = alpha * faiss_score + (1 - alpha) * bm25_score
        doc_scores.append((int(doc_id), combined_score))
        
    doc_scores_sorted = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
    return doc_scores_sorted