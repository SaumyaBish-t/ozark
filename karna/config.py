"""Central configuration loaded from environment / .env file.

All other modules import `settings` from here. Never read env vars directly
elsewhere — keeps the surface area for misconfig small and typed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed config. Reads .env at repo root."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Telegram ----
    telegram_bot_token: Optional[str] = None
    telegram_allowed_user_ids: str = ""  # comma-separated; empty = allow all

    # ---- LLM ----
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"  # 768-dim, ~270MB, fast
    # Generation can take 2-3 min on CPU with a 4-9B model when the prompt
    # is full of recall + market context. 300s is safe for CPU; bump up
    # if you see timeouts, drop down if you have a GPU.
    ollama_timeout_secs: float = 300.0
    openai_api_key: Optional[str] = None
    openai_fallback_model: str = "gpt-4o-mini"

    # ---- Neo4j ----
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "karnapassword"

    # ---- ChromaDB ----
    # "embedded" = in-process (no daemon, default).
    # "http"     = talk to a remote Chroma server (set host/port).
    chroma_mode: str = "embedded"
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # ---- Storage ----
    karna_data_dir: Path = Field(default=REPO_ROOT / "data_local")
    karna_sqlite_path: Path = Field(default=REPO_ROOT / "data_local" / "karna.db")

    # ---- Logging ----
    log_level: str = "INFO"

    # ---- Helpers ----
    @property
    def allowed_user_ids(self) -> list[int]:
        if not self.telegram_allowed_user_ids:
            return []
        return [int(x) for x in self.telegram_allowed_user_ids.split(",") if x.strip()]

    def ensure_dirs(self) -> None:
        """Create data dirs on startup so SQLite/etc don't fail on first write."""
        self.karna_data_dir.mkdir(parents=True, exist_ok=True)
        self.karna_sqlite_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
