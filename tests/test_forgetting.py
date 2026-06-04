"""Ebbinghaus decay smoke tests.

We poke last_access_at backwards by hand to simulate the passage of time
without needing to wait — same trick the real cron will exercise.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from karna.memory.forgetting import ARCHIVE_THRESHOLD, DECAY_RATE_PER_DAY, ForgettingEngine
from karna.memory.session_store import SessionStore


SECONDS_PER_DAY = 86400.0


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    s = SessionStore(db_path=tmp_path / "decay.db")
    yield s
    s.close()


def _backdate(store: SessionStore, turn_id: str, days_ago: float) -> None:
    """Pretend the turn was last accessed `days_ago` days ago."""
    new_ts = time.time() - days_ago * SECONDS_PER_DAY
    conn = store._connect()  # noqa: SLF001
    with conn:
        conn.execute(
            "UPDATE turns SET last_access_at = ? WHERE id = ?", (new_ts, turn_id),
        )


def test_fresh_turns_barely_decay(store: SessionStore) -> None:
    tid = store.store(session_id="s", role="user", content="just now")
    engine = ForgettingEngine(session_store=store)
    engine.run_once()
    row = store._connect().execute("SELECT * FROM turns WHERE id=?", (tid,)).fetchone()  # noqa: SLF001
    assert row["retention_score"] > 0.99


def test_old_turn_archives_below_threshold(store: SessionStore) -> None:
    tid = store.store(session_id="s", role="user", content="ancient memory")
    _backdate(store, tid, days_ago=200)  # ~200 days @ rate 1/30 → exp(-6.67) ≈ 0.0013
    engine = ForgettingEngine(session_store=store)
    report = engine.run_once()
    row = store._connect().execute("SELECT * FROM turns WHERE id=?", (tid,)).fetchone()  # noqa: SLF001
    assert row["retention_score"] < ARCHIVE_THRESHOLD
    assert row["scope"] == "archived"
    assert report.archived == 1


def test_pinned_turn_never_decays(store: SessionStore) -> None:
    tid = store.store(session_id="s", role="user", content="critical fact", pinned=True)
    _backdate(store, tid, days_ago=365)
    engine = ForgettingEngine(session_store=store)
    report = engine.run_once()
    row = store._connect().execute("SELECT * FROM turns WHERE id=?", (tid,)).fetchone()  # noqa: SLF001
    assert row["retention_score"] == 1.0
    assert row["scope"] != "archived"
    assert report.pinned_skipped == 1


def test_decay_formula_roughly_correct(store: SessionStore) -> None:
    """30-day-old memory should land near exp(-1) ≈ 0.368."""
    tid = store.store(session_id="s", role="user", content="month-old")
    _backdate(store, tid, days_ago=30)
    engine = ForgettingEngine(session_store=store)
    engine.run_once()
    row = store._connect().execute("SELECT * FROM turns WHERE id=?", (tid,)).fetchone()  # noqa: SLF001
    expected = math.exp(-DECAY_RATE_PER_DAY * 30)
    assert abs(row["retention_score"] - expected) < 0.01


def test_boost_caps_at_one(store: SessionStore) -> None:
    tid = store.store(session_id="s", role="user", content="boost me")
    engine = ForgettingEngine(session_store=store)
    engine.boost([tid], factor=0.5)
    engine.boost([tid], factor=0.5)
    engine.boost([tid], factor=0.5)
    row = store._connect().execute("SELECT * FROM turns WHERE id=?", (tid,)).fetchone()  # noqa: SLF001
    assert row["retention_score"] <= 1.0 + 1e-9
