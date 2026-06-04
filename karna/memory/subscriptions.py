"""Tracks where to reach each user.

For now: user_id → telegram_chat_id mapping, populated automatically on
the first inbound Telegram message. The scheduler queries this to know
which chats to push proactive quizzes / morning summaries to.

Future channels (Discord, REST webhook) slot in as more columns or a
generic (user_id, channel, target) table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import sqlite3

from karna.config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS subscriptions (
    user_id          TEXT PRIMARY KEY,
    telegram_chat_id INTEGER,
    updated_at       REAL NOT NULL
);
"""


@dataclass
class Subscription:
    user_id: str
    telegram_chat_id: Optional[int]
    updated_at: float


class Subscriptions:
    """SQLite-backed user → channel-target registry."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else settings.karna_sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            self._conn = conn
        return self._conn

    def _init_schema(self) -> None:
        conn = self._connect()
        with conn:
            conn.executescript(SCHEMA)

    # ---------- public API ----------

    def record_telegram(self, user_id: str, chat_id: int) -> None:
        conn = self._connect()
        with conn:
            conn.execute(
                """
                INSERT INTO subscriptions (user_id, telegram_chat_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    telegram_chat_id = excluded.telegram_chat_id,
                    updated_at = excluded.updated_at
                """,
                (user_id, chat_id, time.time()),
            )

    def get(self, user_id: str) -> Optional[Subscription]:
        row = self._connect().execute(
            "SELECT * FROM subscriptions WHERE user_id = ?", (user_id,),
        ).fetchone()
        if not row:
            return None
        return Subscription(
            user_id=row["user_id"],
            telegram_chat_id=row["telegram_chat_id"],
            updated_at=row["updated_at"],
        )

    def all_telegram(self) -> list[Subscription]:
        """Every user with a Telegram chat on file. Used by scheduler jobs."""
        rows = self._connect().execute(
            "SELECT * FROM subscriptions WHERE telegram_chat_id IS NOT NULL"
        ).fetchall()
        return [
            Subscription(
                user_id=r["user_id"],
                telegram_chat_id=r["telegram_chat_id"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
