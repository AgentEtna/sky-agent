"""Conversation memory. Plain SQLite — no database server, no setup.

Set DB_PATH to a mounted volume (e.g. /data/agent.db) to keep memory
across redeploys. Defaults to ./data/agent.db.
"""

import os
import sqlite3

DB_PATH = os.environ.get("DB_PATH", os.path.join("data", "agent.db"))


def _conn() -> sqlite3.Connection:
    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               chat_id TEXT NOT NULL,
               role TEXT NOT NULL,
               content TEXT NOT NULL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    return conn


def add(chat_id: str, role: str, content: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content),
        )


def history(chat_id: str, limit: int = 30) -> list[dict]:
    """Return the last `limit` messages for a chat, oldest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def clear(chat_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
