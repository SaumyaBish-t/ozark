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
from karna.config import settings
from karna.memory.memory import build_memory


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
        meta += "[/dim]"

        console.print(meta)
        console.print(Markdown(reply.text))
        console.print()


if __name__ == "__main__":
    run()
