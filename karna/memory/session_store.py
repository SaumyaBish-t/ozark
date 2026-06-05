"""Layer 1 memory — SQLite + FTS5 for keyword conversation search.

Schema:
    turns         (real table, source of truth)
    turns_fts     (FTS5 virtual table, indexed on content; kept in sync via triggers)

Why two tables: FTS5 virtual tables don't support arbitrary columns / typed
queries well, so we keep typed columns (scope, retention_score, timestamps)
in `turns` and just project content into FTS5 for search.

`retention_score` is updated in place by the forgetting engine (Sprint 2).
For Sprint 1 we just initialise it to 1.0 and never decay it.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from karna.config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    user_id         TEXT,
    role            TEXT NOT NULL,            -- 'user' | 'assistant' | 'system'
    content         TEXT NOT NULL,
    scope           TEXT NOT NULL DEFAULT 'session',  -- session|user|agent|org
    retention_score REAL NOT NULL DEFAULT 1.0,
    pinned          INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    last_access_at  REAL NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_turns_session   ON turns(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_turns_scope     ON turns(scope, retention_score);
CREATE INDEX IF NOT EXISTS idx_turns_user      ON turns(user_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts
USING fts5(content, content='turns', content_rowid='rowid');

-- Keep FTS index in lockstep with the source table.
CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE OF content ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO turns_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


@dataclass
class Turn:
    """One row in `turns`. Returned by search() and history()."""
    id: str
    session_id: str
    user_id: Optional[str]
    role: str
    content: str
    scope: str
    retention_score: float
    pinned: bool
    created_at: float
    last_access_at: float

    def to_chat_message(self) -> dict[str, str]:
        """Shape for LLM message lists."""
        return {"role": self.role, "content": self.content}


class SessionStore:
    """SQLite-backed session memory.

    Cheap to instantiate — connection is opened lazily and cached. Safe for
    use from a single thread (sqlite3's default). For the bot, we'll later
    wrap calls in `asyncio.to_thread` since python-telegram-bot is async.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else settings.karna_sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    # ---------- internals ----------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False lets the Telegram bot's worker
            # thread reuse the connection that the main thread opened.
            # Safe because:
            #   - WAL mode allows concurrent readers + serialised writer
            #   - python-telegram-bot processes messages one at a time
            #   - we hold no cross-statement cursors
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._conn = conn
        return self._conn

    def _init_schema(self) -> None:
        conn = self._connect()
        with conn:
            conn.executescript(SCHEMA)

    # ---------- public API ----------

    def store(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        scope: str = "session",
        pinned: bool = False,
        metadata_json: str = "{}",
    ) -> str:
        """Insert a turn. Returns the new turn's id."""
        now = time.time()
        turn_id = str(uuid.uuid4())
        conn = self._connect()
        with conn:
            conn.execute(
                """
                INSERT INTO turns
                    (id, session_id, user_id, role, content, scope,
                     retention_score, pinned, created_at, last_access_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, 1.0, ?, ?, ?, ?)
                """,
                (turn_id, session_id, user_id, role, content, scope,
                 1 if pinned else 0, now, now, metadata_json),
            )
        return turn_id

    def history(self, session_id: str, limit: int = 20) -> list[Turn]:
        """Last N turns for a session, oldest-first (chat order)."""
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM turns
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [_row_to_turn(r) for r in reversed(rows)]

    def search(self, query: str, *, limit: int = 10, scope: Optional[str] = None) -> list[Turn]:
        """Full-text search across all turns. Bumps last_access_at on hits."""
        conn = self._connect()
        if scope:
            rows = conn.execute(
                """
                SELECT t.* FROM turns_fts f
                JOIN turns t ON t.rowid = f.rowid
                WHERE turns_fts MATCH ? AND t.scope = ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, scope, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT t.* FROM turns_fts f
                JOIN turns t ON t.rowid = f.rowid
                WHERE turns_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

        turns = [_row_to_turn(r) for r in rows]
        if turns:
            self._touch_access([t.id for t in turns])
        return turns

    def pin(self, turn_id: str, pinned: bool = True) -> None:
        conn = self._connect()
        with conn:
            conn.execute("UPDATE turns SET pinned = ? WHERE id = ?", (1 if pinned else 0, turn_id))

    def count(self) -> int:
        return self._connect().execute("SELECT COUNT(*) FROM turns").fetchone()[0]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---------- helpers ----------

    def _touch_access(self, turn_ids: Iterable[str]) -> None:
        ids = list(turn_ids)
        if not ids:
            return
        now = time.time()
        placeholders = ",".join("?" for _ in ids)
        conn = self._connect()
        with conn:
            conn.execute(
                f"UPDATE turns SET last_access_at = ? WHERE id IN ({placeholders})",
                (now, *ids),
            )


def _row_to_turn(r: sqlite3.Row) -> Turn:
    return Turn(
        id=r["id"],
        session_id=r["session_id"],
        user_id=r["user_id"],
        role=r["role"],
        content=r["content"],
        scope=r["scope"],
        retention_score=r["retention_score"],
        pinned=bool(r["pinned"]),
        created_at=r["created_at"],
        last_access_at=r["last_access_at"],
    )
