"""Karna entry point.

Usage:
    python main.py cli                  # interactive terminal chat
    python main.py telegram             # run Telegram bot (needs TELEGRAM_BOT_TOKEN)
    python main.py scheduler            # APScheduler (decay + review checks)
    python main.py seed-graph           # seed Neo4j with the trading concept graph
    python main.py decay                # one-shot Ebbinghaus decay sweep
    python main.py fetch TICKER [RSI]   # smoke-test a market fetch (e.g. fetch RELIANCE RSI)
    python main.py health               # check Ollama / Neo4j / SQLite are reachable
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

    if cmd == "scheduler":
        from karna.scheduler.cron import run as run_scheduler
        run_scheduler()
        return 0

    if cmd == "decay":
        from karna.memory.forgetting import ForgettingEngine
        engine = ForgettingEngine()
        report = engine.run_once()
        print(report)
        return 0

    if cmd == "fetch":
        if len(sys.argv) < 3:
            print("usage: python main.py fetch TICKER [INDICATOR...]")
            return 2
        ticker = sys.argv[2].upper()
        wanted = [a.lower() for a in sys.argv[3:]]
        from karna.data.router import get_router
        from karna.agent.market import format_indicator, format_quote
        from karna.trading import indicators
        r = get_router()
        try:
            q = r.get_quote(ticker)
            print(format_quote(q))
        except Exception as e:
            print(f"quote failed: {e}")
        if wanted:
            try:
                df = r.get_ohlcv(ticker, days=120)
                for ind in wanted:
                    print(format_indicator(ind, indicators.compute(ind, df)))
            except Exception as e:
                print(f"indicator fetch failed: {e}")
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
