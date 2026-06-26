# news-intelligence-system-genai

## Fusion Formula

The retrieval system combines FAISS (dense) and BM25 (sparse) scores. Here's how:

**Step 1 — Normalizing FAISS scores**

FAISS returns L2 distances, so a lower value means a closer (more relevant) match. Because the embedding model (`all-MiniLM-L6-v2`) produces unit-normalized vectors, L2 distances are mathematically bounded between 0 and 2. This lets us both invert and normalize in one step:

$$S^i_{f,\text{norm}} = 1 - \frac{d^i_f}{2}$$

Distance 0 (perfect match) maps to 1.0, distance 2 (completely opposite vectors) maps to 0.0. This is not arbitrary — it works because the vectors are guaranteed to be on the unit hypersphere.

**Step 2 — Normalizing BM25 scores**

BM25 scores don't have a fixed upper bound, so standard min-max is used:

$$S^i_{b,\text{norm}} = \frac{S^i_b - S_{b,\min}}{S_{b,\max} - S_{b,\min} + \epsilon}$$

**Step 3 — Adaptive weight**

Instead of a fixed alpha, the weight is derived from variance of each set of scores-The intuition is if a retriever's scores are spread out (high variance), it's making meaningful distinctions and should be trusted more.

$$\sigma_f^2 = \mathrm{Var}(S^i_{f,\text{norm}}), \qquad \sigma_b^2 = \mathrm{Var}(S^i_{b,\text{norm}})$$

$$\alpha = \frac{\sigma_f^2}{\sigma_f^2 + \sigma_b^2 + \epsilon}$$

**Step 4 — Missing documents**

Documents that appear in only one result set get a score of 0 from the other retriever.

**Final formula:**

$$S^i = \alpha \left(1 - \frac{d^i_f}{2}\right) + (1-\alpha)\left(\frac{S^i_b - S_{b,\min}}{S_{b,\max} - S_{b,\min} + \epsilon}\right)$$

where $\epsilon$ is a small constant to prevent division by zero.

**Why this is different from RRF:**

RRF computes scores purely from rank. It completely ignores how close a document actually is — a document ranked 1st with distance 0.1 and one ranked 1st with distance 1.9 are treated identically. My formula keeps the actual score magnitudes and uses them directly, so a document that scores 0.06 in FAISS (far from query) correctly gets a low final score even if it's the "best" in the batch. The adaptive alpha also means the formula adjusts per query — So for any Query if BM25's Keyword Search is Performing better then it would get a better weight compared FAISS while calculating the Hybrid SCore.
---

## Chunking Design

The data is news articles from NewsAPI — plain text, mostly paragraph-structured, no HTML or rich formatting. Chukning Idea is Simple, Break down textual data into paragraphs and then sentences, then form chunks based on the chunk threshold and how similar the next sentence to be added is to the current one.

```
INPUT: article_text, chunk_threshold, similarity_threshold

Split text into paragraphs on newlines

FOR each paragraph:
    Split paragraph into sentences (on .!?)
    Encode all sentences using the embedding model
    
    current_chunk = sentences[0]
    current_embedding = embeddings[0]
    
    FOR i = 1 to len(sentences):
        sim = cosine_similarity(embeddings[i], current_embedding)
        
        IF char_length(current_chunk + sentences[i]) <= chunk_threshold
           AND sim > similarity_threshold:
            append sentences[i] to current_chunk
            update current_embedding = mean(current_embedding, embeddings[i])
        ELSE:
            save current_chunk
            start new chunk with sentences[i]
    
    save final current_chunk

RETURN all chunks
```

Paragraph boundaries are always treated as hard splits — sentences from different paragraphs never get merged regardless of similarity.

**What this gets wrong:**

- A sentence like "He announced the decision today." has almost no semantic overlap with "The minister was addressing parliament." even though they're about the same thing. These will get split into separate chunks when they should stay together.
- Only works on plain text. Anything with HTML tags, markdown, tables, or code blocks will produce garbage chunks because the structure gets flattened into the text.
- Works badly on academic or legal text where a single sentence can be 3 lines long and a "paragraph" can span a whole page.
- Semantic relationship is only considered for two adjecent sentences only even if there might be more different sentences in the paragraphs that are closer semantically but not by position.

---

## Validation Strategy


**Faithfulness** — does the answer contradict or go beyond the source documents?

Uses `cross-encoder/nli-deberta-v3-small`, a Natural Language Inference classifier. It takes a `(premise, hypothesis)` pair and classifies it as entailment, neutral, or contradiction. For each sentence $s$ in the answer, we check whether any retrieved chunk $c$ entails it:

$$F = \frac{1}{|A|} \sum_{s \in A} \mathbb{1}\left[\max_{c \in C} P(\text{entailment} \mid c, s) > 0.5\right]$$

this model is completely separate from the one generating the answer. It's a discriminative classifier trained on NLI datasets, not a generative language model. It can't "agree with itself" because it wasn't involved in writing the answer.

**Relevance** — did the retrieved chunks actually relate to the question?

Cosine similarity between the question embedding and each chunk embedding, averaged across all retrieved chunks:

$$R = \frac{1}{k}\sum_{i=1}^{k} \cos(E_q, E_{c_i})$$

No LLM involved just the same embedding model used by the retriever. This measures whether the retrieval step found topically related content.

---

## Abstention Rule

When the top fusion score is below a threshold, the system returns "I don't have enough information" instead of generating an answer.

Let $s^* = \max_i S^i$ be the highest fusion score among retrieved documents.

$$s^* < \tau \Rightarrow \text{abstain}$$

where $\tau = 0.45$.

The threshold is set at 0.45 because after applying absolute FAISS normalization and BM25 stopword filtering, a genuinely relevant document scores above 0.5 consistently. Scores below 0.45 indicate the best retrieved document had no meaningful semantic or keyword overlap with the query.

---

## LLM Model Choice

Using `Qwen/Qwen2.5-7B-Instruct` via the HuggingFace Inference API.

Honestly, this wasn't a deeply deliberate choice — the project had a lot of moving parts and the priority was getting something that worked via the free HF API without needing Ollama set up. Qwen2.5-7B-Instruct  instruction-tuned so it follows the system prompt, it handles the "only answer from context" instruction reliably, and it's freely available on the HF.

The tradeoff is obvious — it's a remote API call with latency and rate limits, and more. The plan was always to switch to a local Ollama model once the rest of the system was stable. 


## Fine-tuning Memo

Not filling this in yet — the anomaly component isn't done and this requires more thought about whether fine-tuning makes sense for this use case vs continuing with RAG. Will revisit.
