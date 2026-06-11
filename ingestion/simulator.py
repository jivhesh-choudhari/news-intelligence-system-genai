# Each article fetch is treated as a reading session. Simulate realistic reading sessions synthetically — generate both human-like and bot-like sessions programmatically

import random
import uuid
import sqlite3
from datetime import datetime, timedelta

class ReadingSessionSimulator:
    def __init__(self, db_path):
        try:
            self.conn = sqlite3.connect(db_path)
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            return
        c = self.conn.cursor()
        c.execute('''
                    CREATE TABLE IF NOT EXISTS reading_sessions (
                        event_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        article_id TEXT,
                        event_name TEXT,
                        timestamp TEXT
                    );
                ''')
        self.conn.commit()
        
    def simulate_session(self, type: str = "human"):
        session_id = uuid.uuid4().hex

        # weights
        if type == "bot":
            w = [
                10,     # no of articles
                1,      # avg events per article
                0.01,   # min time interval multiplier
            ]
            event_weights = [1, 1, 8]  # page_view, scroll, click
        else:
            w = [
                2,      # no of articles
                3,      # avg events per article
                2.0,    # min time interval multiplier
            ]
            event_weights = [1, 5, 2]

        # weighted attributes
        no_articles = random.randint(1, 3) * w[0]
        avg_event_per_article = random.randint(1, 3) * w[1]
        min_time_interval = random.randint(10, 30) * w[2]

        start_time = datetime.now()
        recent_event_time = start_time
        c = self.conn.cursor()
        c.execute(
            "SELECT article_id FROM articles ORDER BY RANDOM() LIMIT ?",
            (int(no_articles),)
        )
        article_ids = c.fetchall()

        for article_id_tuple in article_ids:
            article_id = article_id_tuple[0]
            events_for_article = max(
                1,
                int(random.uniform(avg_event_per_article * 0.5, avg_event_per_article * 1.0))
            )
            for _ in range(events_for_article):
                event_id = uuid.uuid4().hex
                event_name = random.choices(
                    ['page_view', 'scroll', 'click'],
                    weights=event_weights,
                    k=1
                )[0]
                interval = random.uniform(min_time_interval, min_time_interval * 3)
                recent_event_time += timedelta(seconds=interval)
                c.execute(
                    """
                    INSERT INTO reading_sessions (event_id, session_id, article_id, event_name, timestamp) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        session_id,
                        article_id,
                        event_name,
                        recent_event_time.isoformat()
                    )
                )

        self.conn.commit()
            
# if __name__ == "__main__":
#     simulator = ReadingSessionSimulator('news.db')
#     simulator.simulate_session(type="human")
#     simulator.simulate_session(type="bot")