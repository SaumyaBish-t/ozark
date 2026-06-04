"""Scheduled jobs that read/write the memory + graph layers.

Each function is meant to be safely callable in isolation — useful for
manual invocation (`python -m karna.scheduler.retention_checker decay`)
and for unit testing.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from karna.connectors.outbound import send_telegram
from karna.memory.forgetting import ForgettingEngine
from karna.memory.subscriptions import Subscriptions
from karna.skills.quiz import generate_question, pick_concept_for_review


log = logging.getLogger(__name__)


# ---------- decay sweep ----------

def daily_decay_job() -> None:
    """Apply Ebbinghaus decay to every unpinned, non-archived turn."""
    engine = ForgettingEngine()
    report = engine.run_once()
    log.info("daily decay finished: %s", report)


# ---------- review nudge ----------

def _open_graph_store():
    """Best-effort connect to Neo4j. Returns None when unavailable —
    review jobs just skip in that case."""
    try:
        from karna.memory.graph_store import GraphStore
        gs = GraphStore()
        with gs._driver.session() as s:  # noqa: SLF001
            s.run("RETURN 1").consume()
        return gs
    except Exception as e:
        log.warning("Neo4j unavailable in scheduler: %s", e)
        return None


def hourly_review_job() -> None:
    """For each subscribed user, push at most one quiz if anything is due."""
    graph = _open_graph_store()
    if graph is None:
        return

    subs = Subscriptions()
    users = subs.all_telegram()
    log.info("hourly review scan over %d subscribed users", len(users))

    for sub in users:
        try:
            picked = pick_concept_for_review(graph, sub.user_id)
        except Exception as e:
            log.exception("pick_concept_for_review for %s failed: %s", sub.user_id, e)
            continue
        if picked is None:
            continue  # nothing due

        concept, last_level = picked
        try:
            quiz = generate_question(concept, last_level)
        except Exception as e:
            log.exception("generate_question failed for %s/%s: %s",
                          sub.user_id, concept.id, e)
            continue

        msg = (
            f"⏰ Quick review: *{concept.name}*\n\n"
            f"{quiz.question}\n\n"
            f"(Reply when you can — I'll grade and update your progress.)"
        )
        sent = send_telegram(sub.user_id, msg)
        if sent:
            log.info("sent quiz to user=%s concept=%s", sub.user_id, concept.id)


# ---------- CLI for manual triggers ----------

def main(argv: Optional[list[str]] = None) -> int:
    """Run a single job from the command line:

        python -m karna.scheduler.retention_checker decay
        python -m karna.scheduler.retention_checker review
    """
    logging.basicConfig(level=logging.INFO)
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: retention_checker.py [decay|review]")
        return 2
    name = argv[0]
    if name == "decay":
        daily_decay_job()
    elif name == "review":
        hourly_review_job()
    else:
        print(f"unknown job: {name}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
