"""Semantic consolidation.

Cognitive-science model: when you rehearse one memory, semantically related
memories also get strengthened (the "consolidation" effect). FSFM 2026
shows this matters for retention quality over long horizons.

Implementation: on each retrieval hit, find the 5 nearest neighbours in
ChromaDB and bump their retention_score by 15% in SQLite.

Boundary: we don't consolidate across users. The current MVP doesn't store
user_id in Chroma metadata for every row yet — once we do (Sprint 5 user
scope), filter neighbours by user_id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from karna.memory.memory import Memory
from karna.memory.forgetting import ForgettingEngine


log = logging.getLogger(__name__)


NEIGHBORS_PER_HIT = 5
BOOST_FACTOR = 0.15


@dataclass
class ConsolidationReport:
    triggered_for: int = 0   # turns we tried to consolidate around
    neighbors_boosted: int = 0


class Consolidator:
    """Wraps Memory + ForgettingEngine. Cheap to instantiate.

    Call `boost_neighbors([turn_ids])` whenever a retrieval surfaced
    those turns to the LLM. We deliberately don't auto-boost on raw
    storage — only on *access*, mirroring spaced-repetition theory.
    """

    def __init__(
        self,
        memory: Memory,
        forgetting: Optional[ForgettingEngine] = None,
        *,
        k: int = NEIGHBORS_PER_HIT,
        factor: float = BOOST_FACTOR,
    ):
        self.memory = memory
        self.forgetting = forgetting or ForgettingEngine(session_store=memory.session)
        self.k = k
        self.factor = factor

    def boost_neighbors(self, turn_ids: Iterable[str]) -> ConsolidationReport:
        """For each accessed turn, find its k semantic neighbours and bump
        their retention. Skips silently if the semantic store is offline.
        """
        report = ConsolidationReport()
        if self.memory.semantic is None:
            return report

        all_neighbor_ids: list[str] = []
        for tid in turn_ids:
            report.triggered_for += 1
            try:
                neighbors = self.memory.semantic.nearest_neighbors(tid, k=self.k)
            except Exception as e:
                log.debug("neighbor lookup failed for %s: %s", tid, e)
                continue
            for nid in neighbors:
                if nid not in all_neighbor_ids:
                    all_neighbor_ids.append(nid)

        if all_neighbor_ids:
            report.neighbors_boosted = self.forgetting.boost(all_neighbor_ids, factor=self.factor)
        return report
