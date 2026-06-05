"""Local data cache.

Two tables in the same SQLite Karna already uses:

    market_quotes      — latest snapshot per (ticker, source), TTL'd
    market_ohlcv       — per (ticker, interval, date) row; we store one
                         row per bar so partial overlaps reuse what's there

Why SQLite again: it's already running, transactional, queryable, and the
volume here (a few thousand rows per stock over years) is trivially small.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from karna.config import settings
from karna.data.base import OHLCV_COLS, Quote


log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS market_quotes (
    ticker      TEXT NOT NULL,
    source      TEXT NOT NULL,
    payload     TEXT NOT NULL,    -- json-serialised Quote
    fetched_at  REAL NOT NULL,
    PRIMARY KEY (ticker, source)
);

CREATE TABLE IF NOT EXISTS market_ohlcv (
    ticker   TEXT NOT NULL,
    interval TEXT NOT NULL,
    bar_date TEXT NOT NULL,        -- ISO yyyy-mm-dd
    open     REAL NOT NULL,
    high     REAL NOT NULL,
    low      REAL NOT NULL,
    close    REAL NOT NULL,
    volume   INTEGER NOT NULL,
    PRIMARY KEY (ticker, interval, bar_date)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date
    ON market_ohlcv(ticker, interval, bar_date);
"""


# Quote TTL during market hours vs after — both kept short so we don't
# serve stale prices but long enough to absorb a burst of CLI commands.
QUOTE_TTL_OPEN_SECS = 60        # 1 min while market is live
QUOTE_TTL_CLOSED_SECS = 60 * 60 # 1 hour after-hours


class DataCache:
    """SQLite-backed cache. Stateless beyond the connection."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else settings.karna_sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # See session_store._connect for rationale on check_same_thread=False.
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            self._conn = conn
        return self._conn

    def _init_schema(self) -> None:
        conn = self._connect()
        with conn:
            conn.executescript(SCHEMA)

    # ---------- quotes ----------

    def get_quote(
        self,
        ticker: str,
        *,
        source: Optional[str] = None,
        market_open: bool = False,
        max_age: Optional[float] = None,
    ) -> Optional[Quote]:
        """Return the most-recent cached quote if fresh enough."""
        ttl = max_age if max_age is not None else (
            QUOTE_TTL_OPEN_SECS if market_open else QUOTE_TTL_CLOSED_SECS
        )
        cutoff = time.time() - ttl
        conn = self._connect()
        if source:
            row = conn.execute(
                "SELECT * FROM market_quotes WHERE ticker=? AND source=? AND fetched_at >= ?",
                (ticker.upper(), source, cutoff),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM market_quotes WHERE ticker=? AND fetched_at >= ?
                ORDER BY fetched_at DESC LIMIT 1
                """,
                (ticker.upper(), cutoff),
            ).fetchone()
        if not row:
            return None
        data = json.loads(row["payload"])
        return Quote(**data)

    def put_quote(self, quote: Quote) -> None:
        conn = self._connect()
        with conn:
            conn.execute(
                """
                INSERT INTO market_quotes (ticker, source, payload, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker, source) DO UPDATE SET
                    payload = excluded.payload,
                    fetched_at = excluded.fetched_at
                """,
                (quote.ticker, quote.source, json.dumps(asdict(quote)), quote.timestamp),
            )

    # ---------- ohlcv ----------

    def get_ohlcv(
        self,
        ticker: str,
        *,
        interval: str = "1d",
        days: int = 90,
    ) -> Optional[pd.DataFrame]:
        """Return cached bars if we have at least `days` rows for the ticker.

        Returns None when we'd have to top up from a provider anyway —
        forces the router to refresh rather than serve incomplete data.
        """
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT bar_date AS Date, open AS Open, high AS High, low AS Low,
                   close AS Close, volume AS Volume
            FROM market_ohlcv
            WHERE ticker = ? AND interval = ?
            ORDER BY bar_date DESC
            LIMIT ?
            """,
            (ticker.upper(), interval, days),
        ).fetchall()
        if len(rows) < days:
            return None
        df = pd.DataFrame([dict(r) for r in rows])
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)

    def put_ohlcv(self, ticker: str, df: pd.DataFrame, *, interval: str = "1d") -> int:
        """Upsert bars. Returns number of rows written."""
        if df is None or df.empty:
            return 0
        missing = [c for c in OHLCV_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

        rows: list[tuple] = []
        for _, r in df.iterrows():
            d = pd.Timestamp(r["Date"]).date().isoformat()
            rows.append((ticker.upper(), interval, d,
                         float(r["Open"]), float(r["High"]), float(r["Low"]),
                         float(r["Close"]), int(r["Volume"])))

        conn = self._connect()
        with conn:
            conn.executemany(
                """
                INSERT INTO market_ohlcv
                    (ticker, interval, bar_date, open, high, low, close, volume)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker, interval, bar_date) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low  = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume
                """,
                rows,
            )
        return len(rows)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
