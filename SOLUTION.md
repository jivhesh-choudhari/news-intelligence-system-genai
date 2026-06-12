# news-intelligence-system-genai


### Chunking - Why I chose this approach

The data source for this project is NewsAPI. The articles returned by NewsAPI are already mostly plain text and do not contain a rich HTML structure that could be used for section-aware chunking.
Because of that, I decided not to use embedding-based semantic chunking or any external chunking library. Instead, I implemented a lightweight recursive chunking strategy from scratch that attempts to preserve the natural structure of a news article.
The main goal was to keep related information together.
---

#### Pseudocode

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