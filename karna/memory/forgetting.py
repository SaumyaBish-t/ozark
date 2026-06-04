"""Ebbinghaus forgetting engine.

Every memory has a `retention_score` ∈ [0, 1]. It decays exponentially with
time since the user last accessed it:

    retention = base_strength * exp(-decay_rate * dt_days)

Pinned turns are skipped. Below `archive_threshold` we flip the scope to
`archived` — the memory still exists (research: FadeMem 2026, "45% storage
reduction, no accuracy loss"), but recall queries exclude it by default.

Modulators (some now, some Sprint 6+):
    - access_count   : each recall hit bumps base_strength (spaced repetition)
    - consolidation  : neighbors of accessed memories get a small bump
    - pin            : never decays
    - contradiction  : forced demotion (Sprint 5)

Decay rate is tuned so a turn untouched for ~30 days decays to ~0.37
(the half-life knob is `DECAY_RATE_PER_DAY`).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from karna.memory.session_store import SessionStore


log = logging.getLogger(__name__)


# Tunables. Externalise to config once we have empirical data (Sprint 5+).
DECAY_RATE_PER_DAY = 1.0 / 30.0       # half-life ≈ 30 days
ARCHIVE_THRESHOLD = 0.20              # below this → archive
SECONDS_PER_DAY = 86400.0


@dataclass
class DecayReport:
    """Stats from one decay sweep — surface in the CLI / cron logs."""
    scanned: int = 0
    decayed: int = 0
    archived: int = 0
    pinned_skipped: int = 0
    already_archived_skipped: int = 0

    def __str__(self) -> str:
        return (
            f"DecayReport(scanned={self.scanned}, decayed={self.decayed}, "
            f"archived={self.archived}, pinned_skipped={self.pinned_skipped})"
        )


class ForgettingEngine:
    """Applies decay to every eligible turn in the session store."""

    def __init__(
        self,
        session_store: Optional[SessionStore] = None,
        *,
        decay_rate: float = DECAY_RATE_PER_DAY,
        archive_threshold: float = ARCHIVE_THRESHOLD,
    ):
        self.store = session_store or SessionStore()
        self.decay_rate = decay_rate
        self.archive_threshold = archive_threshold

    def run_once(self, now: Optional[float] = None) -> DecayReport:
        """Apply decay to every non-pinned, non-archived turn.

        Returns a report. Safe to call repeatedly — idempotent in the sense
        that calling it twice in quick succession is the same as calling it
        once (decay measured from last_access_at, not from last decay run).
        """
        now = now or time.time()
        report = DecayReport()
        conn = self.store._connect()  # noqa: SLF001 — engine owns the store
        rows = conn.execute(
            """
            SELECT id, retention_score, last_access_at, pinned, scope
            FROM turns
            """
        ).fetchall()

        updates: list[tuple[float, str, str]] = []  # (new_score, new_scope, id)

        for r in rows:
            report.scanned += 1
            if r["pinned"]:
                report.pinned_skipped += 1
                continue
            if r["scope"] == "archived":
                report.already_archived_skipped += 1
                continue

            dt_days = max(0.0, (now - r["last_access_at"]) / SECONDS_PER_DAY)
            new_score = r["retention_score"] * math.exp(-self.decay_rate * dt_days)
            new_score = max(0.0, min(1.0, new_score))

            new_scope = r["scope"]
            if new_score < self.archive_threshold:
                new_scope = "archived"
                report.archived += 1

            # Only write if something actually changed (avoids no-op churn
            # on first run when dt_days ≈ 0).
            if (abs(new_score - r["retention_score"]) > 1e-6
                    or new_scope != r["scope"]):
                updates.append((new_score, new_scope, r["id"]))
                report.decayed += 1

        if updates:
            with conn:
                conn.executemany(
                    "UPDATE turns SET retention_score = ?, scope = ? WHERE id = ?",
                    updates,
                )
        log.info("forgetting: %s", report)
        return report

    def boost(self, turn_ids: list[str], *, factor: float = 0.15) -> int:
        """Multiplicatively bump retention for the given turns (capped at 1.0).

        Used by:
            - the consolidation engine (5 nearest neighbors of accessed turn)
            - explicit user signals ("/pin")
            - the LLM-as-judge when a turn was actually useful

        Returns the number of rows updated. `factor=0.15` = +15%.
        """
        if not turn_ids:
            return 0
        conn = self.store._connect()  # noqa: SLF001
        placeholders = ",".join("?" for _ in turn_ids)
        with conn:
            cur = conn.execute(
                f"""
                UPDATE turns
                SET retention_score = MIN(1.0, retention_score * (1 + ?)),
                    last_access_at = ?
                WHERE id IN ({placeholders}) AND scope != 'archived'
                """,
                (factor, time.time(), *turn_ids),
            )
            return cur.rowcount
