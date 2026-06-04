"""Layer 3 memory — Neo4j knowledge graph.

Sprint 1 scope:
    - Seed the trading concept tree (idempotent via MERGE).
    - Track per-user comprehension state on PREREQ paths.
    - Find missing prerequisites for a target concept.

Later sprints will add:
    - Trade nodes linked to concepts (Sprint 4)
    - Strategy nodes + outcomes (Sprint 5)
    - Temporal edges with timestamps for forgetting interaction (Sprint 2)
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from neo4j import GraphDatabase, Driver

from karna.config import settings
from karna.trading.concepts import Concept


log = logging.getLogger(__name__)


# Comprehension levels stored on (:User)-[:UNDERSTANDS]->(:Concept).
COMPREHENSION_LEVELS = ("initial", "reviewed", "solid", "mastered")


class GraphStore:
    """Wraps a Neo4j driver. Safe to keep one instance per process."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._driver: Driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    # ---------- schema / seed ----------

    def _ensure_constraints(self) -> None:
        """Concept ids must be unique. Run once on seed."""
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT concept_id IF NOT EXISTS "
                  "FOR (c:Concept) REQUIRE c.id IS UNIQUE")
            s.run("CREATE CONSTRAINT user_id IF NOT EXISTS "
                  "FOR (u:User) REQUIRE u.id IS UNIQUE")

    def seed(self, concepts: Iterable[Concept]) -> tuple[int, int]:
        """Idempotently insert the concept tree. Returns (#nodes, #edges).

        Uses MERGE so repeated calls don't create duplicates.
        """
        self._ensure_constraints()
        concepts = list(concepts)
        n_nodes = 0
        n_edges = 0
        with self._driver.session() as s:
            for c in concepts:
                s.run(
                    """
                    MERGE (n:Concept {id: $id})
                    SET n.name = $name,
                        n.category = $category,
                        n.depth = $depth,
                        n.summary = $summary
                    """,
                    id=c.id, name=c.name, category=c.category,
                    depth=c.depth, summary=c.summary,
                )
                n_nodes += 1
            for c in concepts:
                for p in c.prereqs:
                    s.run(
                        """
                        MATCH (child:Concept {id: $child_id})
                        MATCH (parent:Concept {id: $parent_id})
                        MERGE (child)-[:REQUIRES]->(parent)
                        """,
                        child_id=c.id, parent_id=p,
                    )
                    n_edges += 1
        return n_nodes, n_edges

    # ---------- comprehension state ----------

    def upsert_user(self, user_id: str, name: Optional[str] = None) -> None:
        with self._driver.session() as s:
            s.run(
                """
                MERGE (u:User {id: $id})
                SET u.name = COALESCE($name, u.name)
                """,
                id=user_id, name=name,
            )

    def set_comprehension(
        self, user_id: str, concept_id: str, level: str
    ) -> None:
        """Record what the user has understood about a concept."""
        if level not in COMPREHENSION_LEVELS:
            raise ValueError(f"level must be one of {COMPREHENSION_LEVELS}, got {level!r}")
        with self._driver.session() as s:
            s.run(
                """
                MERGE (u:User {id: $uid})
                MERGE (c:Concept {id: $cid})
                MERGE (u)-[r:UNDERSTANDS]->(c)
                SET r.level = $level,
                    r.updated_at = timestamp()
                """,
                uid=user_id, cid=concept_id, level=level,
            )

    def get_comprehension(self, user_id: str, concept_id: str) -> Optional[str]:
        with self._driver.session() as s:
            rec = s.run(
                """
                MATCH (:User {id: $uid})-[r:UNDERSTANDS]->(:Concept {id: $cid})
                RETURN r.level AS level
                """,
                uid=user_id, cid=concept_id,
            ).single()
            return rec["level"] if rec else None

    # ---------- prerequisite traversal ----------

    def missing_prereqs(
        self,
        user_id: str,
        concept_id: str,
        *,
        considered_known: tuple[str, ...] = ("solid", "mastered"),
    ) -> list[str]:
        """Walk REQUIRES edges from `concept_id` and return ids the user
        hasn't reached at least one of `considered_known` levels on.

        Order: deepest prerequisites first, so the teach loop covers
        foundations before dependents.
        """
        with self._driver.session() as s:
            rows = s.run(
                """
                MATCH path = (target:Concept {id: $cid})-[:REQUIRES*0..]->(prereq:Concept)
                OPTIONAL MATCH (:User {id: $uid})-[r:UNDERSTANDS]->(prereq)
                WITH prereq, r, length(path) AS dist
                WHERE r IS NULL OR NOT r.level IN $known
                RETURN DISTINCT prereq.id AS id, max(dist) AS dist
                ORDER BY dist DESC
                """,
                cid=concept_id, uid=user_id, known=list(considered_known),
            )
            # Exclude the target concept itself — we only want prereqs.
            return [r["id"] for r in rows if r["id"] != concept_id]
