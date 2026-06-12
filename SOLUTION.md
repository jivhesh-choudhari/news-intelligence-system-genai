# news-intelligence-system-genai

## Fusion Formula

Considerations:
1. Since FAISS Scores Are distances (Less the Better) they are need to be inverted.
Considering S as Score I am using Linear Inversion: 
$$\text{Inverted Score} = {Smax - S}$$

2. Both Scores need Normalized at same Scale, For simplicity i will use Min Max Normalization
Formula: $$S_{\text{norm}} = \frac{S - S_{\text{min}}}{S_{\text{max}} - S_{\text{min}}}$$

3. Finally I am using Liner Interpolation so that Symetic scores have more weight compared to Sparse Scores.
Formula: $$S_{\text{hybrid}} = \alpha \cdot S_{\text{faiss}} + (1 - \alpha) \cdot S_{\text{bm25}}$$

4. Document Missing in either FAISS or BM25 are given the lowest scores


Finally, for any Document with ID $i$ in Top K Documents, Fusion Score is Derived as:

$$S^i = \alpha \cdot \left(\frac{S_{\text{f,max}} - S^i_f}{S_{\text{f,max}} - S_{\text{f,min}}} \right) + (1 - \alpha) \cdot \left( \frac{S^i_b - S_{\text{b,min}}}{S_{\text{b,max}} - S_{\text{b,min}}} \right)$$

where 

$S^i$ = The final hybrid score for document

$i$.$\alpha$ = The weight assigned to the dense (FAISS) retrieval.

$S^i_f$, $S^i_b$ = The raw FAISS and BM25 scores for document $i$.

$S_{\text{f,max}}$, $S_{\text{f,min}}$ = The highest and lowest raw FAISS scores in the retrieved list.

$S_{\text{b,max}}$, $S_{\text{b,min}}$ = The highest and lowest raw BM25 scores in the retrieved list.

Constrainsts:

$S_{\text{max}} \neq S_{\text{min}}$ 

$k > 1$


## Chunking - Why I chose this approach


- The data source for this project is NewsAPI. The articles returned by NewsAPI are already mostly plain text and do not contain a rich HTML structure that could be used for section-aware chunking.
- Because of that, I decided not to use embedding-based semantic chunking or any external chunking library. Instead, I implemented a lightweight recursive chunking strategy from scratch that attempts to preserve the natural structure of a news article.

```text
INPUT: article_text, chunk_threshold

Split article into paragraphs

FOR each paragraph:
    Split paragraph into sentences
    Create empty current_chunk
    FOR each sentence:
        IF adding sentence keeps chunk size below threshold:
            append sentence to current_chunk
        ELSE:
            save current_chunk
            start new chunk with current sentence
    IF current_chunk is not empty:
        save current_chunk
RETURN all chunks
```