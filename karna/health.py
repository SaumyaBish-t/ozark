"""Lightweight health checks for the local infra Karna depends on.

Run with: `python main.py health`
Useful during onboarding and as a smoke test after `docker compose up -d`.
"""

from __future__ import annotations

import socket
import sqlite3
from typing import Callable

import httpx
from rich.console import Console
from rich.table import Table

from karna.config import settings


console = Console()


def _check_ollama() -> tuple[bool, str]:
    try:
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0)
        r.raise_for_status()
        models = [m.get("name", "?") for m in r.json().get("models", [])]
        if not models:
            return False, "reachable but no models pulled (run: docker exec -it karna-ollama ollama pull llama3.1:8b)"
        target = settings.ollama_model
        present = any(m.startswith(target.split(":")[0]) for m in models)
        return present, f"models: {', '.join(models)}"
    except Exception as e:
        return False, f"unreachable: {e}"


def _check_neo4j() -> tuple[bool, str]:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
        driver.close()
        return True, f"connected at {settings.neo4j_uri}"
    except Exception as e:
        return False, f"unreachable: {e}"


def _check_chromadb() -> tuple[bool, str]:
    try:
        with socket.create_connection((settings.chroma_host, settings.chroma_port), timeout=2.0):
            pass
        return True, f"port {settings.chroma_port} open"
    except Exception as e:
        return False, f"unreachable: {e}"


def _check_sqlite() -> tuple[bool, str]:
    try:
        settings.ensure_dirs()
        conn = sqlite3.connect(str(settings.karna_sqlite_path))
        conn.execute("SELECT sqlite_version()")
        conn.close()
        return True, f"writable at {settings.karna_sqlite_path}"
    except Exception as e:
        return False, f"failed: {e}"


CHECKS: dict[str, Callable[[], tuple[bool, str]]] = {
    "Ollama":   _check_ollama,
    "Neo4j":    _check_neo4j,
    "ChromaDB": _check_chromadb,
    "SQLite":   _check_sqlite,
}


def run_health_checks() -> bool:
    table = Table(title="Karna health check")
    table.add_column("Service")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True
    for name, check in CHECKS.items():
        ok, detail = check()
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(name, status, detail)
        all_ok = all_ok and ok

    console.print(table)
    return all_ok
