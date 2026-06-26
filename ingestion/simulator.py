
import sqlite3, random, uuid, time, threading
from datetime import datetime, timedelta

class ReadingSessionSimulator:
    def __init__(self, db_path="session.db"):
        self.conn=sqlite3.connect(db_path)
        c=self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS reading_sessions(
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            session_type TEXT,
            article_id TEXT,
            event_name TEXT,
            timestamp TEXT
        )""")
        self.conn.commit()

    def _emit_event(self,sid,stype,article,event,ts):
        self.conn.execute("INSERT INTO reading_sessions(session_id,session_type,article_id,event_name,timestamp) VALUES(?,?,?,?,?)",
                          (sid,stype,article,event,ts.isoformat()))
        self.conn.commit()

    def _next_human(self,last):
        r=random.random()
        if last=="page_view":
            return "scroll" if r<0.8 else "click"
        if last=="scroll":
            return "scroll" if r<0.6 else ("click" if r<0.85 else "page_view")
        return "page_view"

    def _next_bot(self,last):
        r=random.random()
        if last=="page_view":
            return "click" if r<0.7 else "page_view"
        if last=="click":
            return "page_view"
        return "click"

    def simulate_human_session(self):
        sid=uuid.uuid4().hex
        ts=datetime.now()
        event="page_view"
        for _ in range(random.randint(15,40)):
            art=f"article_{random.randint(1,100)}"
            self._emit_event(sid,"human",art,event,ts)
            gap=random.uniform(3,18)
            ts+=timedelta(seconds=gap)
            event=self._next_human(event)
        return sid

    def simulate_bot_session(self):
        sid=uuid.uuid4().hex
        ts=datetime.now()
        event="page_view"
        for _ in range(random.randint(40,120)):
            art=f"article_{random.randint(1,100)}"
            self._emit_event(sid,"bot",art,event,ts)
            gap=random.uniform(0.05,0.5)
            ts+=timedelta(seconds=gap)
            event=self._next_bot(event)
        return sid

    def stream_events(self,kind=None):
        kind=kind or random.choice(["human","bot"])
        sid=uuid.uuid4().hex
        ts=datetime.now()
        event="page_view"
        while True:
            art=f"article_{random.randint(1,100)}"
            self._emit_event(sid,kind,art,event,ts)
            if kind=="human":
                sleep=random.uniform(2,8)
                event=self._next_human(event)
                ts+=timedelta(seconds=sleep)
            else:
                sleep=random.uniform(0.05,0.4)
                event=self._next_bot(event)
                ts+=timedelta(seconds=sleep)
            time.sleep(sleep)

    def stream_sessions(self,max_active=3):
        active=[]
        while True:
            while len(active)<max_active:
                stype="bot" if not any(s["type"]=="bot" for s in active) else random.choices(["human","bot"],weights=[3,1])[0]
                active.append({"id":uuid.uuid4().hex,"type":stype,"event":"page_view","ts":datetime.now(),"remaining":random.randint(20,60) if stype=="human" else random.randint(50,120)})
            s=random.choice(active)
            art=f"article_{random.randint(1,100)}"
            self._emit_event(s["id"],s["type"],art,s["event"],s["ts"])
            if s["type"]=="human":
                sleep=random.uniform(2,7)
                s["event"]=self._next_human(s["event"])
            else:
                sleep=random.uniform(0.05,0.3)
                s["event"]=self._next_bot(s["event"])
            s["ts"]+=timedelta(seconds=sleep)
            s["remaining"]-=1
            if s["remaining"]<=0:
                active.remove(s)
            time.sleep(random.uniform(0.2,1.0))

# if __name__=="__main__":
#     sim=ReadingSessionSimulator("session.db")
#     # sim.simulate_human_session()
#     # sim.simulate_bot_session()
#     # sim.stream_events("human")
#     sim.stream_sessions()


def start_background_streaming(mode: str = "sessions", db_path: str = "session.db"):
    """
    mode: 'sessions' streams interleaved multi-session data (default)
          'events'   streams a single continuous session
    """
    sim = ReadingSessionSimulator(db_path)

    def run():
        if mode == "events":
            sim.stream_events()
        else:
            sim.stream_sessions()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread
