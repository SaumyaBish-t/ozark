"""Karna entry point.

Usage:
    python main.py cli         # interactive terminal chat (works without Telegram token)
    python main.py telegram    # run Telegram bot (needs TELEGRAM_BOT_TOKEN in .env)
    python main.py seed-graph  # seed Neo4j with the trading concept graph
    python main.py health      # check Ollama / Neo4j / SQLite are reachable
"""

from __future__ import annotations

import sys

from karna.config import settings


HELP = __doc__


def main() -> int:
    settings.ensure_dirs()

    if len(sys.argv) < 2:
        print(HELP)
        return 0

    cmd = sys.argv[1].lower()

    if cmd in {"-h", "--help", "help"}:
        print(HELP)
        return 0

    if cmd == "cli":
        from karna.connectors.cli import run as run_cli
        run_cli()
        return 0

    if cmd == "telegram":
        from karna.connectors.telegram_bot import run as run_telegram
        run_telegram()
        return 0

    if cmd == "seed-graph":
        from karna.memory.graph_store import GraphStore
        from karna.trading.concepts import CONCEPT_TREE
        gs = GraphStore()
        n_nodes, n_edges = gs.seed(CONCEPT_TREE)
        print(f"Seeded {n_nodes} concept nodes and {n_edges} prerequisite edges.")
        gs.close()
        return 0

    if cmd == "health":
        from karna.health import run_health_checks
        ok = run_health_checks()
        return 0 if ok else 1

    print(f"Unknown command: {cmd}\n")
    print(HELP)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
