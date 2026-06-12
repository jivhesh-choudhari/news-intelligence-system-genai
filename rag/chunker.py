import re
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from nltk.tokenize import word_tokenize

class Chunker:
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        chunk_threshold=300,
        index_path="faiss.index",
        metadata_path="metadata.db",
        bm25_path="bm25.db"
    ):
        self.model = SentenceTransformer(model_name)
        self.chunk_threshold = chunk_threshold
        self.faiss_path = index_path

        # SQLite for chunks
        self.conn = sqlite3.connect(metadata_path)
        self.create_metadata_table()

        # BM25 storage
        self.bm25_conn = sqlite3.connect(bm25_path)
        self.create_bm25_tables()
        self.index = None
        
    def create_metadata_table(self):
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT
            )
        """)
        self.conn.commit()

    def create_bm25_tables(self):
        c = self.bm25_conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS bm25_docs (
            chunk_id INTEGER,
            tokens TEXT,
            length INTEGER
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS bm25_df (
            term TEXT PRIMARY KEY,
            df INTEGER
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS bm25_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)

        self.bm25_conn.commit()

    def get_sentences_from_paragraph(self, paragraph)->list[str]:
        '''
            Returns All sentences from a given Paragraph
        '''
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        return sentences

    def get_chunks_from_text(self, text)->list[str]:
        '''
            Gives a List of Sentences as output, For each paragraph in a Text
        ''' 
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
        print(f"Produced {len(chunks)} Chunks for {len(paragraphs)} Paragraphs")
        return chunks
    
    def make_tokens(self, text):
        '''
            Reduces Sentence Chunks to individual Tokens for creating index
        ''' 
        tokens = set()
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalnum()]
        return tokens
    
    def make_embeddings(self, chunks):
        '''
            Generates Embeddings 
        '''
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        return embeddings, dimension
    
    def build_bm25(self, chunks, chunk_ids):
        '''
            Generates DF (Document Frequency for a Token) and List of all Tokens For That Chunk
        '''
        df = defaultdict(int)
        doc_tokens = {}

        for chunk_id, chunk in zip(chunk_ids, chunks):
            tokens = self.make_tokens(chunk)
            doc_tokens[chunk_id] = tokens

            for token in set(tokens):
                df[token] += 1

        return doc_tokens, df
      
    def save(self, chunks):
        '''
            Saves Three things:
            1. Metadata to Each Sentence Chunk
            2. BM25 Index
            3. Embedding Vectors 
        ''' 
        self.cursor = self.conn.cursor()
        
        #    Metadata    #
        for chunk in chunks:
            self.cursor.execute("INSERT INTO chunks (content) VALUES (?)", (chunk,))
        self.conn.commit()
        last_id = self.cursor.lastrowid 
        if not last_id:
            ids = np.arange(1, len(chunks) + 1)
        else:
            ids = np.arange(abs(last_id - len(chunks)) + 1, last_id + 1)
        print("Saved Metadata")
            
        #   BM25   #
        doc_tokens, df = self.build_bm25(chunks, ids)
        bm25_cursor = self.bm25_conn.cursor()
        for chunk_id, tokens in doc_tokens.items():
            bm25_cursor.execute("""
                INSERT INTO bm25_docs VALUES (?, ?, ?)
            """, (chunk_id, " ".join(tokens), len(tokens)))
        
        # store DF
        for term, freq in df.items():
            bm25_cursor.execute("""
                INSERT OR REPLACE INTO bm25_df VALUES (?, ?)
            """, (term, freq))

        bm25_cursor.execute("""
            INSERT OR REPLACE INTO bm25_meta VALUES (?, ?)
        """, ("N", str(len(chunks))))
        self.bm25_conn.commit()
        print("Saved BM25 Index")
        
        
        #    Embeddings    #
        embeddings, dimension = self.make_embeddings(chunks)
        base_index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIDMap(base_index)
        self.index.add_with_ids(embeddings, ids)
        faiss.write_index(self.index, self.faiss_path)
        print("Saved Embeddings")
        
    def run(self, text):
        chunks = self.get_chunks_from_text(text)
        self.save(chunks)

# if __name__ == "__main__":
#     text = """The Reserve Bank of India (RBI) announced a new liquidity framework today. Analysts say the move could improve credit availability for small businesses. However, several economists warned that the long-term effects remain uncertain. What impact will this have on inflation over the next 12 months?\n\nMarkets reacted positively to the announcement. The NIFTY 50 gained 1.2%, while the SENSEX climbed nearly 800 points during afternoon trading. Traders described the rally as "unexpectedly strong" given recent volatility. Some investors remained cautious, citing concerns about global growth.\nMeanwhile, technology companies continued to invest heavily in artificial intelligence. Microsoft reported increased spending on data-center infrastructure, while Google expanded its AI research initiatives. These investments are expected to exceed $50 billion by 2027. Could this trigger a new wave of competition across the industry?\nThe report also highlighted cybersecurity concerns. Attackers frequently exploit misconfigured cloud services, weak authentication mechanisms, and outdated software packages. Security teams are encouraged to perform regular audits, monitor unusual activity, and apply patches promptly. Failure to do so may expose sensitive customer information.\nConsumer sentiment improved slightly in June despite persistent inflationary pressures. Survey respondents expressed optimism about wage growth but remained concerned about housing costs. Economists noted that household spending patterns have shifted considerably since 2024. Further data will be released next quarter."""
    
#     chunker = Chunker()
#     chunker.run(text)