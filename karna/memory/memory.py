"""Memory facade — agent-facing entry point.

Composes the three storage layers so the Agent doesn't have to know which
fact lives where:

    Layer 1 (SessionStore / SQLite FTS5)
        - source of truth for turn metadata + retention scores
        - keyword search

    Layer 2 (SemanticStore / ChromaDB)
        - vector embeddings keyed by the same turn_id as Layer 1
        - semantic search

    Layer 3 (GraphStore / Neo4j) — accessed directly by the agent for
        concept reasoning, not through this facade.

Public surface:
    store(...)     → dual-writes to L1 + L2
    recall(query)  → unions FTS5 + semantic, deduped, ranked
    history(...)   → recent turns from L1 (chat continuity)

Sprint 2 ships the basics. Sprint 5 adds the cross-session promotion policy.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from karna.memory.session_store import SessionStore, Turn


log = logging.getLogger(__name__)


@dataclass
class Recall:
    """One memory retrieval hit."""
    turn_id: str
    content: str
    role: str
    session_id: str
    score: float          # combined relevance (higher = better)
    source: str           # "fts" | "semantic" | "both"
    retention: float      # current retention_score (from L1)
    distance: Optional[float] = None  # semantic distance, if applicable


class Memory:
    """Owns the L1+L2 storage layers. Safe singleton per process."""

    def __init__(
        self,
        session_store: Optional[SessionStore] = None,
        semantic_store=None,  # SemanticStore | None — optional so agent runs degraded if Chroma is down
    ):
        self.session = session_store or SessionStore()
        self.semantic = semantic_store  # may be None
        self._semantic_enabled = semantic_store is not None

    # ---------- writes ----------

    def store(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        scope: str = "session",
        pinned: bool = False,
    ) -> str:
        """Persist a turn. Returns the turn_id (shared across L1 and L2)."""
        created_at = time.time()
        turn_id = self.session.store(
            session_id=session_id,
            role=role,
            content=content,
            user_id=user_id,
            scope=scope,
            pinned=pinned,
        )
        if self._semantic_enabled:
            try:
                self.semantic.add(
                    turn_id=turn_id,
                    content=content,
                    session_id=session_id,
                    role=role,
                    user_id=user_id,
                    scope=scope,
                    created_at=created_at,
                )
            except Exception as e:
                # Don't fail the whole turn if Chroma hiccups — L1 already has it.
                log.warning("semantic_store.add failed for turn %s: %s", turn_id, e)
        return turn_id

    # ---------- reads ----------

    def history(self, session_id: str, limit: int = 20) -> list[Turn]:
        """Recent turns for chat continuity."""
        return self.session.history(session_id, limit=limit)

    def recall(
        self,
        query: str,
        *,
        k: int = 6,
        user_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> list[Recall]:
        """Union of FTS5 + semantic retrieval, deduped on turn_id and
        re-ranked. Falls back to FTS-only when ChromaDB is down."""
        # ---- L1 keyword hits ----
        fts_hits = self.session.search(query, limit=k, scope=scope)
        results: dict[str, Recall] = {}
        for rank, t in enumerate(fts_hits):
            results[t.id] = Recall(
                turn_id=t.id,
                content=t.content,
                role=t.role,
                session_id=t.session_id,
                # FTS bm25-ish ranking — convert rank to a normalised score.
                score=1.0 - (rank / max(len(fts_hits), 1)),
                source="fts",
                retention=t.retention_score,
            )

        # ---- L2 semantic hits ----
        if self._semantic_enabled:
            try:
                sem_hits = self.semantic.search(query, k=k, scope=scope, user_id=user_id)
                for hit in sem_hits:
                    tid = hit["id"]
                    # Cosine distance ranges 0..2. 0 = identical → score 1.
                    dist = hit.get("distance") or 1.0
                    sem_score = max(0.0, 1.0 - (dist / 2.0))
                    if tid in results:
                        # Boost things both layers agree on.
                        results[tid].source = "both"
                        results[tid].score = max(results[tid].score, sem_score) + 0.15
                        results[tid].distance = dist
                    else:
                        # Pull the L1 row so we have retention + role.
                        turn = self._fetch_turn(tid)
                        if turn is None:
                            continue
                        results[tid] = Recall(
                            turn_id=tid,
                            content=turn.content,
                            role=turn.role,
                            session_id=turn.session_id,
                            score=sem_score,
                            source="semantic",
                            retention=turn.retention_score,
                            distance=dist,
                        )
            except Exception as e:
                log.warning("semantic_store.search failed, falling back to FTS only: %s", e)

        ranked = sorted(results.values(), key=lambda r: r.score, reverse=True)
        return ranked[:k]

    # ---------- maintenance ----------

    def count(self) -> int:
        return self.session.count()

    # ---------- internals ----------

    def _fetch_turn(self, turn_id: str) -> Optional[Turn]:
        """Pull a single turn from SQLite by id. Cheap, indexed."""
        conn = self.session._connect()  # noqa: SLF001 — facade is the owner
        row = conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
        if not row:
            return None
        from karna.memory.session_store import _row_to_turn
        return _row_to_turn(row)


# ---------- factory: build a Memory with graceful fallback ----------

def build_memory(*, with_semantic: bool = True) -> Memory:
    """Construct a Memory instance, wiring up the semantic store if Chroma
    is reachable. If not, returns a degraded Memory that still works on FTS5.

    Use this from connectors / scripts so they don't have to know how the
    layers are composed.
    """
    session = SessionStore()
    semantic = None
    if with_semantic:
        try:
            from karna.memory.semantic_store import SemanticStore
            semantic = SemanticStore()
            # Probe — calling count() forces the HTTP client to handshake.
            semantic.count()
        except Exception as e:
            log.warning("ChromaDB unavailable, running with FTS5 only: %s", e)
            semantic = None
    return Memory(session_store=session, semantic_store=semantic)
