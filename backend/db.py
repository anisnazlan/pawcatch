"""Database layer: one SQLite file (data/wildwatch.db), built into Python, no setup needed.

On Render's free plan this file is reset whenever the server restarts; the sample
reports are recreated automatically on startup (see render.yaml).
"""
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "wildwatch.db"))


def query(q: str, params=(), fetch: str = "all"):
    """Run one statement. fetch = 'all' | 'one' | 'none'. Returns dict rows."""
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(q, params)
        if fetch == "all":
            return [dict(r) for r in cur.fetchall()]
        if fetch == "one":
            r = cur.fetchone()
            return dict(r) if r else None
        return cur.rowcount


def query_many(q: str, rows):
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.executemany(q, rows)


def init_db():
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    query("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        lat REAL NOT NULL, lng REAL NOT NULL,
        animal TEXT NOT NULL, incident TEXT NOT NULL,
        urgency TEXT NOT NULL, agency TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'received',
        confidence REAL, ai_mode TEXT,
        photo TEXT, note TEXT,
        resolved_at TEXT, simulated INTEGER DEFAULT 0
    )""", fetch="none")
