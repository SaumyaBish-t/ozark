"""Terminal chat — fastest way to iterate without a Telegram token.

    python main.py cli

Commands inside the REPL:
    /quit, /exit       — leave
    /new               — start a fresh session id
    /history           — dump current session history
    /concepts <text>   — show which concepts the agent would detect in <text>
"""

from __future__ import annotations

import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from karna.agent.core import Agent, AgentReply, detect_concepts, new_session_id
from karna.agent.market import format_indicator, format_quote
from karna.config import settings
from karna.data.router import get_router as get_data_router
from karna.memory.memory import build_memory
from karna.trading import indicators


console = Console()


def _try_graph_store():
    """Best-effort Neo4j connect. None if it's not up — agent still works."""
    try:
        from karna.memory.graph_store import GraphStore
        gs = GraphStore()
        # Trip the connection lazily.
        with gs._driver.session() as s:  # noqa: SLF001 — intentional probe
            s.run("RETURN 1").consume()
        return gs
    except Exception as e:
        console.print(f"[yellow]Neo4j unavailable, running without concept graph:[/yellow] {e}")
        return None


def run(user_id: Optional[str] = None) -> None:
    settings.ensure_dirs()
    user_id = user_id or "local-user"

    memory = build_memory()
    graph = _try_graph_store()
    agent = Agent(memory=memory, graph_store=graph)

    session_id = new_session_id()

    console.print(Panel.fit(
        f"[bold]Karna CLI[/bold]\nsession: {session_id}\nuser:    {user_id}\n"
        f"model:   {settings.ollama_model}\n"
        f"Type /quit to exit.",
        border_style="cyan",
    ))

    while True:
        try:
            msg = console.input("[bold green]you ▸[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye.[/dim]")
            break
        if not msg:
            continue

        # ---- meta commands ----
        if msg in {"/quit", "/exit"}:
            break
        if msg == "/new":
            session_id = new_session_id()
            console.print(f"[cyan]new session:[/cyan] {session_id}")
            continue
        if msg == "/history":
            for t in memory.history(session_id, limit=50):
                console.print(f"[dim]{t.role}:[/dim] {t.content}")
            continue
        if msg.startswith("/concepts "):
            hits = detect_concepts(msg[len("/concepts "):])
            console.print(f"detected: {hits or '(none)'}")
            continue
        if msg.startswith("/recall "):
            q = msg[len("/recall "):]
            for r in memory.recall(q, k=5, user_id=user_id):
                console.print(
                    f"[dim]{r.source} {r.score:.2f} ret={r.retention:.2f}:[/dim] "
                    f"({r.role}) {r.content[:140]}"
                )
            continue

        # ---- market data commands (deterministic, no LLM) ----
        if msg.startswith("/price "):
            ticker = msg[len("/price "):].strip().upper()
            try:
                q = get_data_router().get_quote(ticker)
                console.print(format_quote(q))
            except Exception as e:
                console.print(f"[red]price fetch failed:[/red] {e}")
            continue
        if any(msg.startswith(f"/{ind} ") for ind in ("rsi", "macd", "bb", "atr", "sma", "ema")):
            head, _, rest = msg[1:].partition(" ")
            ticker = rest.strip().upper()
            try:
                df = get_data_router().get_ohlcv(ticker, days=120)
                result = indicators.compute(head, df)
                console.print(format_indicator(head, result))
            except Exception as e:
                console.print(f"[red]{head} fetch failed:[/red] {e}")
            continue
        if msg.startswith("/chart "):
            ticker = msg[len("/chart "):].strip().upper()
            try:
                df = get_data_router().get_ohlcv(ticker, days=15).tail(10)
                console.print(f"[bold]{ticker}[/bold] last 10 sessions:")
                for _, r in df.iterrows():
                    console.print(
                        f"  {r['Date'].date()}  "
                        f"O {r['Open']:>9.2f}  H {r['High']:>9.2f}  "
                        f"L {r['Low']:>9.2f}  C {r['Close']:>9.2f}  "
                        f"V {int(r['Volume']):>12,}"
                    )
            except Exception as e:
                console.print(f"[red]chart fetch failed:[/red] {e}")
            continue

        # ---- normal turn ----
        try:
            reply: AgentReply = agent.handle(msg, session_id=session_id, user_id=user_id)
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")
            continue

        meta = f"[dim]intent={reply.intent} via {reply.matched_rule}"
        if reply.mentioned_concepts:
            meta += f" | concepts={','.join(reply.mentioned_concepts)}"
        if reply.prereq_focus:
            meta += f" | prereqs={','.join(reply.prereq_focus[:3])}"
        if reply.recalled:
            meta += f" | recalled={len(reply.recalled)}"
        if reply.market_context is not None:
            mc = reply.market_context
            tag = f"market:{mc.ticker}"
            if mc.indicator_results:
                tag += f"({','.join(mc.indicator_results.keys())})"
            meta += f" | {tag}"
        meta += "[/dim]"

        console.print(meta)
        console.print(Markdown(reply.text))
        console.print()


if __name__ == "__main__":
    run()
