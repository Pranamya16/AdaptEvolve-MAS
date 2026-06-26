import os
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

# Absolute path so every caller uses the same file regardless of cwd
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptevolve.db")


def init_db(db_path: str = DB_PATH) -> None:
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            username        TEXT NOT NULL,
            goal            TEXT NOT NULL,
            started_at      TEXT NOT NULL,
            finished_at     TEXT,
            max_cycles      INTEGER,
            population_size INTEGER,
            num_generations INTEGER,
            final_score     REAL,
            n_cycles_run    INTEGER,
            status          TEXT DEFAULT 'running',
            best_code       TEXT,
            score_trajectory TEXT,
            dimension_scores TEXT,
            mechanic_log    TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            chat_id     TEXT PRIMARY KEY,
            username    TEXT NOT NULL,
            title       TEXT NOT NULL,
            messages    TEXT NOT NULL DEFAULT '[]',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_username ON sessions(username)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_started ON sessions(started_at DESC)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chat_username ON chat_sessions(username)")
    # Safe migration: add message_ratings column if it doesn't exist yet
    try:
        con.execute("ALTER TABLE chat_sessions ADD COLUMN message_ratings TEXT DEFAULT '{}'")
    except sqlite3.OperationalError:
        pass  # column already exists
    con.commit()
    con.close()


def new_session_id() -> str:
    return str(uuid.uuid4())


def insert_session(
    session_id: str,
    username: str,
    goal: str,
    max_cycles: int,
    population_size: int,
    num_generations: int,
    db_path: str = DB_PATH,
) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """INSERT INTO sessions
           (session_id, username, goal, started_at, max_cycles, population_size,
            num_generations, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'running')""",
        (
            session_id,
            username,
            goal,
            datetime.now(timezone.utc).isoformat(),
            max_cycles,
            population_size,
            num_generations,
        ),
    )
    con.commit()
    con.close()


def update_session(session_id: str, db_path: str = DB_PATH, **fields) -> None:
    if not fields:
        return
    # Serialize list/dict values to JSON strings
    serialized = {}
    for k, v in fields.items():
        serialized[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    set_clause = ", ".join(f"{k} = ?" for k in serialized)
    values = list(serialized.values()) + [session_id]
    con = sqlite3.connect(db_path)
    con.execute(f"UPDATE sessions SET {set_clause} WHERE session_id = ?", values)
    con.commit()
    con.close()


def get_user_sessions(username: str, limit: int = 50, db_path: str = DB_PATH) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT session_id, goal, started_at, finished_at, final_score,
                  n_cycles_run, status, max_cycles
           FROM sessions WHERE username = ?
           ORDER BY started_at DESC LIMIT ?""",
        (username, limit),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def upsert_chat_session(
    chat_id: str,
    username: str,
    title: str,
    messages: list,
    db_path: str = DB_PATH,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(db_path)
    con.execute(
        """INSERT INTO chat_sessions (chat_id, username, title, messages, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(chat_id) DO UPDATE SET
               messages   = excluded.messages,
               updated_at = excluded.updated_at""",
        (chat_id, username, title, json.dumps(messages), now, now),
    )
    con.commit()
    con.close()


def get_user_chat_sessions(username: str, limit: int = 30, db_path: str = DB_PATH) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT chat_id, title, updated_at FROM chat_sessions
           WHERE username = ? ORDER BY updated_at DESC LIMIT ?""",
        (username, limit),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_chat_session_messages(chat_id: str, db_path: str = DB_PATH) -> list:
    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT messages FROM chat_sessions WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    con.close()
    if not row:
        return []
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return []


def save_message_rating(
    chat_id: str, msg_idx: int, rating: int, db_path: str = DB_PATH
) -> None:
    """Store a thumbs-up (1) or thumbs-down (-1) for a single assistant message."""
    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT message_ratings FROM chat_sessions WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    if not row:
        con.close()
        return
    try:
        ratings = json.loads(row[0] or "{}")
    except (json.JSONDecodeError, TypeError):
        ratings = {}
    ratings[str(msg_idx)] = rating
    con.execute(
        "UPDATE chat_sessions SET message_ratings = ? WHERE chat_id = ?",
        (json.dumps(ratings), chat_id),
    )
    con.commit()
    con.close()


def get_message_ratings(chat_id: str, db_path: str = DB_PATH) -> dict:
    """Return {str(msg_idx): rating} for a chat session."""
    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT message_ratings FROM chat_sessions WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    con.close()
    if not row:
        return {}
    try:
        return json.loads(row[0] or "{}") or {}
    except (json.JSONDecodeError, TypeError):
        return {}


def get_session_detail(session_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    con.close()
    if row is None:
        return None
    d = dict(row)
    for field in ("score_trajectory", "dimension_scores", "mechanic_log"):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d
