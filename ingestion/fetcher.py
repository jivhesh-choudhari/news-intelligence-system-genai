import requests
import threading
import time
import os
from dotenv import load_dotenv, find_dotenv

try:
    from .storage import ArticleStorage
except ImportError:
    from storage import ArticleStorage

load_dotenv(find_dotenv())


class NewsFetcher:
    def __init__(self, country="us", api_key=None):
        self.country = country
        self.api_key = api_key or os.getenv("NEWS_API_KEY")

    def fetch(self) -> list:
        url = f"https://newsapi.org/v2/top-headlines?country={self.country}&apiKey={self.api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("articles", [])


def start_background_polling(interval: int = 600):
    storage = ArticleStorage()
    fetcher = NewsFetcher()

    def poll_loop():
        while True:
            try:
                articles = fetcher.fetch()
                storage.save(articles)
                print(f"[fetcher] Stored {len(articles)} articles")
            except Exception as e:
                print(f"[fetcher] Error: {e}")
            time.sleep(interval)

    thread = threading.Thread(target=poll_loop, daemon=True)
    thread.start()
    return thread


