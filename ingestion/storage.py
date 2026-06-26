import sqlite3
import uuid
from datetime import datetime


class ArticleStorage:
    def __init__(self, db_path="news.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.cursor().execute("""
            CREATE TABLE IF NOT EXISTS articles (
                article_id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                source TEXT,
                published_at TEXT,
                fetched_at TEXT
            )
        """)
        self.conn.commit()

    def save(self, articles: list):
        c = self.conn.cursor()
        for article in articles:
            if not article.get("title") or not article.get("content"):
                continue
            c.execute(
                "INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?, ?, ?)",
                (
                    uuid.uuid4().hex,
                    article["title"],
                    article["content"],
                    article.get("source", {}).get("name", "unknown"),
                    article.get("publishedAt", ""),
                    datetime.now().isoformat(),
                ),
            )
        self.conn.commit()
