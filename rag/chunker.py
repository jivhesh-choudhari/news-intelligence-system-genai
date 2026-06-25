import re
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

class Chunker:
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        chunk_threshold=300,
        similarity_threshold=0.45,
        index_path="faiss.index",
        metadata_path="metadata.db",
        bm25_path="bm25.db"
    ):
        self.model = SentenceTransformer(model_name)
        self.chunk_threshold = chunk_threshold
        self.faiss_path = index_path
        self.similarity_threshold = similarity_threshold 
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
            embeddings = self.model.encode(sentences) 
            current_chunk = sentences[0]
            curr_chunk_embedding = embeddings[0]
            for i in range(1, len(sentences)):
                sentence = sentences[i]
                # Find Similarity between the current chunk and the next sentence
                sim = cosine_similarity(embeddings[i].reshape(1, -1), curr_chunk_embedding.reshape(1, -1)).item()
                # Merging Chunks based on the threshold and similarity
                if len(current_chunk) + len(sentence) + 1 <= self.chunk_threshold and sim > self.similarity_threshold:
                    current_chunk += " " + sentence
                    # Update Chunk Embedding by taking the mean 
                    curr_chunk_embedding = np.mean([curr_chunk_embedding, embeddings[i]], axis=0)
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    curr_chunk_embedding = embeddings[i]
            if current_chunk:
                chunks.append(current_chunk.strip())
        print(f"Produced {len(chunks)} Chunks for {len(paragraphs)} Paragraphs")
        return chunks
    
    def make_tokens(self, text):
        '''
            Reduces Sentence Chunks to individual Tokens for creating index
        ''' 
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        text = ''.join([char for char in text if char not in punctuation])
        text = ' '.join([word for word in text.split() if word not in stop_words])
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalnum()]
        return tokens
    
    def make_embeddings(self, chunks):
        '''
            Generates Embeddings 
        '''
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        return embeddings, dimension
    
    def build_bm25(self, chunks, chunk_ids):
        '''
            Generates DF (Document Frequency for a Token) and List of all Tokens For That Chunk
        '''
        doc_tokens = {}

        for chunk_id, chunk in zip(chunk_ids, chunks):
            tokens = self.make_tokens(chunk)
            doc_tokens[chunk_id] = tokens

        return doc_tokens
      
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
            ids = list(range(1, len(chunks) + 1))
        else:
            ids = list(range(abs(last_id - len(chunks)) + 1, last_id + 1))
        print("Saved Metadata")
            
        #   BM25   #
        doc_tokens = self.build_bm25(chunks, ids)
        bm25_cursor = self.bm25_conn.cursor()
        for chunk_id, tokens in doc_tokens.items():
            bm25_cursor.execute("""
                INSERT INTO bm25_docs VALUES (?, ?, ?)
            """, (chunk_id, " ".join(tokens), len(tokens)))    
        self.bm25_conn.commit()
        print("Saved BM25 Index")
        
        #    Embeddings    #
        embeddings, dimension = self.make_embeddings(chunks)
        base_index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIDMap(base_index)
        self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))
        faiss.write_index(self.index, self.faiss_path)
        print("Saved Embeddings")
        
    def run(self, text):
        chunks = self.get_chunks_from_text(text)
        self.save(chunks)

if __name__ == "__main__":
    text = 'Artificial intelligence is changing how people interact with technology every day. From recommendation systems to virtual assistants, machine learning models are becoming increasingly common! Researchers continue to explore new architectures that improve efficiency and accuracy. Meanwhile, penguins in Antarctica spend months adapting to extreme weather conditions.\n\n\n\nThe old lighthouse stood on the edge of the cliff, overlooking the restless sea. Waves crashed against the rocks below, sending sprays of water into the air! Despite decades of storms, the structure remained remarkably resilient. How many sailors had relied on its guiding light over the years?\n\nA small café opened near the train station last month. Customers quickly became fond of its freshly baked pastries and aromatic coffee! Some visitors came to work quietly on their laptops, while others gathered with friends. What makes a place feel welcoming and memorable?\n\n\nDeep within the rainforest, countless species coexist in a delicate balance. Brightly colored birds dart between the trees while insects hum in the background! Environmental scientists monitor these ecosystems to better understand biodiversity. Can conservation efforts keep pace with the challenges posed by climate change?' 
    
    chunker = Chunker()
    chunker.run(text)