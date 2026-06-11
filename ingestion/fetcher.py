import requests
import sqlite3
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

class NewsFetcher:
    def __init__(self, country, news_api_key):
        self.country = country
        self.news_api_key = news_api_key
        self.conn = sqlite3.connect('news.db')
        c = self.conn.cursor()
        c.execute('''
                  CREATE TABLE IF NOT EXISTS articles (
                        article_id TEXT PRIMARY KEY,
                        title TEXT,
                        content TEXT,
                        source TEXT,
                        published_at TEXT,
                        fetched_at TEXT
                    );
                ''')
        self.conn.commit()

    def fetch_news(self):
        url = f"https://newsapi.org/v2/top-headlines?country={self.country}&apiKey={self.news_api_key}"
        response = requests.get(url)
        return response.json()

    def store_news(self, data):
        c = self.conn.cursor()
        for article in data['articles']:
            c.execute("INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, article['title'], article['content'], article['source']['name'], article['publishedAt'], datetime.now()))
        self.conn.commit()
    
    # for debug
    def print_recent_news(self):
        c = self.conn.cursor()
        c.execute("SELECT * FROM articles ORDER BY fetched_at DESC LIMIT 5")
        rows = c.fetchall()
        for row in rows:
            print(row)

# if __name__ == "__main__":
#     news_api_key = os.getenv('NEWS_API_KEY')
#     fetcher = NewsFetcher('us', news_api_key)
#     news_data = fetcher.fetch_news()
#     fetcher.store_news(news_data)
#     fetcher.print_recent_news()


