import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sqlite3

class Chunker:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_threshold=300, index_path="faiss.index", metadata_path="metadata.db"):
        self.model = SentenceTransformer(model_name)
        self.chunk_threshold = chunk_threshold
        self.conn = sqlite3.connect(metadata_path)
        self.index = None
        self.create_metadata_table()
        
    def create_metadata_table(self):
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT
            )
        """)
        self.conn.commit()

    def get_sentences_from_paragraph(self, paragraph)->list[str]:
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        return sentences

    def get_chunks_from_text(self, text)->list[str]:
        paragraphs = re.split(r'\n+', text.strip()) #Splitting the text into paragraphs based on \n or \n\n
        chunks = []
        for paragraph in paragraphs: # Prevents Cross-Paragraph Sentence Merging/Splitting
            sentences = self.get_sentences_from_paragraph(paragraph)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= self.chunk_threshold:
                    current_chunk += " " + sentence
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def build_and_save_embeddings(self, chunks, path="faiss.index"):
         
        self.cursor = self.conn.cursor()
        for chunk in chunks:
            self.cursor.execute("INSERT INTO chunks (content) VALUES (?)", (chunk,))
        self.conn.commit()
        last_id = self.cursor.lastrowid 
        if not last_id:
            ids = np.arange(1, len(chunks) + 1)
        else:
            ids = np.arange(abs(last_id - len(chunks)) + 1, last_id + 1)
        
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]

        base_index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIDMap(base_index)
        self.index.add_with_ids(embeddings, ids)
        faiss.write_index(self.index, path)

    #for debug
    def print_chunks(self, chunks):
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk}\n")
            
    def print_faiss_embeddings(self, path="faiss.index"):
        index = faiss.read_index(path)
        print("FAISS Index Loaded:")
        print(f"Number of vectors: {index.ntotal}")
        print(f"Dimension: {index.d}")

# if __name__ == "__main__":
#     text = """The Reserve Bank of India (RBI) announced a new liquidity framework today. Analysts say the move could improve credit availability for small businesses. However, several economists warned that the long-term effects remain uncertain. What impact will this have on inflation over the next 12 months?\n\nMarkets reacted positively to the announcement. The NIFTY 50 gained 1.2%, while the SENSEX climbed nearly 800 points during afternoon trading. Traders described the rally as "unexpectedly strong" given recent volatility. Some investors remained cautious, citing concerns about global growth.\nMeanwhile, technology companies continued to invest heavily in artificial intelligence. Microsoft reported increased spending on data-center infrastructure, while Google expanded its AI research initiatives. These investments are expected to exceed $50 billion by 2027. Could this trigger a new wave of competition across the industry?\nThe report also highlighted cybersecurity concerns. Attackers frequently exploit misconfigured cloud services, weak authentication mechanisms, and outdated software packages. Security teams are encouraged to perform regular audits, monitor unusual activity, and apply patches promptly. Failure to do so may expose sensitive customer information.\nConsumer sentiment improved slightly in June despite persistent inflationary pressures. Survey respondents expressed optimism about wage growth but remained concerned about housing costs. Economists noted that household spending patterns have shifted considerably since 2024. Further data will be released next quarter."""
    
#     chunker = Chunker()
#     chunks = chunker.get_chunks_from_text(text)
#     # chunker.print_chunks(chunks)
#     chunker.build_and_save_embeddings(chunks)