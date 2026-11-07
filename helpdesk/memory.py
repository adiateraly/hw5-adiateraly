"""USER memory: preferences that persist across conversations, keyed by user_id.

Separate from the LangGraph checkpointer, which stores CONVERSATION state per thread.
"""
import sqlite3
from contextlib import closing
from pathlib import Path


class UserMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS preferences (
                user_id TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL,
                PRIMARY KEY (user_id, name))""")

    def get_all(self, user_id: str) -> dict:
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT name, value FROM preferences WHERE user_id = ?",
                              (user_id,)).fetchall()
        return dict(rows)

    def set(self, user_id: str, name: str, value: str) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO preferences (user_id, name, value) VALUES (?, ?, ?) "
                       "ON CONFLICT(user_id, name) DO UPDATE SET value = excluded.value",
                       (user_id, name, value))
