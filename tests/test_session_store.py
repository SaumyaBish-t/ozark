"""Smoke tests for the SQLite FTS5 session store.

Runs against a temp DB so it doesn't touch your real data_local/karna.db.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from karna.memory.session_store import SessionStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    s = SessionStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def test_store_and_history(store: SessionStore) -> None:
    sid = "s1"
    store.store(session_id=sid, role="user", content="what is RSI?")
    store.store(session_id=sid, role="assistant", content="RSI is Relative Strength Index.")
    hist = store.history(sid)
    assert [t.role for t in hist] == ["user", "assistant"]
    assert "RSI" in hist[1].content


def test_fts_search_finds_term(store: SessionStore) -> None:
    sid = "s2"
    store.store(session_id=sid, role="user", content="explain bollinger bands please")
    store.store(session_id=sid, role="user", content="what is MACD")
    hits = store.search("bollinger")
    assert len(hits) == 1
    assert "bollinger" in hits[0].content.lower()


def test_search_respects_scope(store: SessionStore) -> None:
    store.store(session_id="s3", role="user", content="trading question one", scope="session")
    store.store(session_id="s3", role="user", content="trading question two", scope="user")
    user_hits = store.search("trading", scope="user")
    assert len(user_hits) == 1
    assert user_hits[0].scope == "user"


def test_count(store: SessionStore) -> None:
    assert store.count() == 0
    store.store(session_id="s", role="user", content="x")
    store.store(session_id="s", role="assistant", content="y")
    assert store.count() == 2
