"""Layer 2 memory — ChromaDB for dense semantic search.

Lives alongside the SQLite session store. Same `turn_id` identifies the
same fact in both layers — that's the join key.

Why both layers?
    - SQLite FTS5 is great when the user uses the same words. Misses synonyms.
    - ChromaDB embeds meaning, finds things lexically different but related
      ("how do I size my position?" → retrieves a turn about "risk per trade").
    - Together they cover keyword search + semantic search without picking one.

Embeddings come from local Ollama (nomic-embed-text by default). Keeping
embedding inference in-process so we don't depend on a cloud service.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Silence ChromaDB's posthog telemetry — it ships with a SDK version that
# raises "capture() takes 1 positional argument but 3 were given" warnings
# on every run, including per-add. Several hooks need to be disabled because
# the env var alone misses CollectionAddEvent and friends.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings

# Belt and braces: kill the telemetry loggers that print despite the flag.
for _name in (
    "chromadb.telemetry",
    "chromadb.telemetry.product",
    "chromadb.telemetry.product.posthog",
):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

# Last resort — the "Failed to send telemetry event" messages come from
# print() inside chromadb's Posthog wrapper, not from the logger. Stub out
# the capture method directly so it can't print anything.
try:
    from chromadb.telemetry.product.posthog import Posthog as _ChromaPosthog
    _ChromaPosthog.capture = lambda *a, **kw: None
except Exception:
    pass

from karna.config import settings
from karna.llm.ollama import OllamaClient, get_client as get_ollama


log = logging.getLogger(__name__)


COLLECTION_NAME = "karna_turns"


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Adapter so Chroma calls our local Ollama for embeddings.

    Chroma's interface: __call__(input: list[str]) -> Embeddings.
    """

    def __init__(self, llm: Optional[OllamaClient] = None):
        self._llm = llm or get_ollama()

    def __call__(self, input: list[str]) -> Embeddings:  # noqa: A002 — name forced by chromadb interface
        # OllamaClient.embed returns list[list[float]] which is exactly Embeddings.
        return self._llm.embed(input)


class SemanticStore:
    """Karna's ChromaDB wrapper.

    Defaults to **embedded** mode — ChromaDB runs in-process, writing to
    `settings.karna_data_dir / "chroma"`. No daemon, no Docker container,
    no network hop. Set CHROMA_MODE=http (and CHROMA_HOST/PORT) in .env
    if you want to talk to a remote Chroma server instead.

    `add()` mirrors a row from SessionStore — same turn_id used as the Chroma id.
    """

    def __init__(
        self,
        *,
        mode: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        llm: Optional[OllamaClient] = None,
    ):
        chosen_mode = (mode or settings.chroma_mode).lower()
        chroma_cfg = ChromaSettings(anonymized_telemetry=False)
        if chosen_mode == "http":
            self._client = chromadb.HttpClient(
                host=host or settings.chroma_host,
                port=port or settings.chroma_port,
                settings=chroma_cfg,
            )
        else:
            # Embedded persistent client — writes to disk, no server needed.
            data_dir = settings.karna_data_dir / "chroma"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(data_dir), settings=chroma_cfg)

        self._embed = OllamaEmbeddingFunction(llm=llm)
        # get_or_create_collection is idempotent — safe across restarts.
        self._coll = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- public API ----------

    def add(
        self,
        *,
        turn_id: str,
        content: str,
        session_id: str,
        role: str,
        user_id: Optional[str] = None,
        scope: str = "session",
        created_at: Optional[float] = None,
    ) -> None:
        """Insert a turn. ChromaDB embeds the content via our Ollama function."""
        metadata = {
            "session_id": session_id,
            "role": role,
            "scope": scope,
        }
        if user_id:
            metadata["user_id"] = user_id
        if created_at is not None:
            metadata["created_at"] = created_at

        self._coll.add(
            ids=[turn_id],
            documents=[content],
            metadatas=[metadata],
        )

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        scope: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Return top-k semantically similar turns.

        Each result: {"id": turn_id, "content": str, "metadata": dict, "distance": float}.
        Distance is cosine — lower means more similar.
        """
        where: dict = {}
        if scope:
            where["scope"] = scope
        if user_id:
            where["user_id"] = user_id

        result = self._coll.query(
            query_texts=[query],
            n_results=k,
            where=where or None,
        )

        out: list[dict] = []
        # Chroma returns lists-of-lists keyed by query (we only sent one query).
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, tid in enumerate(ids):
            out.append({
                "id": tid,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            })
        return out

    def nearest_neighbors(self, turn_id: str, k: int = 5) -> list[str]:
        """Find ids of the k turns most semantically similar to `turn_id`.

        Used by the consolidation engine to boost related memories on access.
        Excludes the source turn itself from the result.
        """
        # Fetch the source doc by id, then re-query with its content.
        got = self._coll.get(ids=[turn_id], include=["documents"])
        docs = got.get("documents") or []
        if not docs:
            return []
        source_doc = docs[0]

        result = self._coll.query(
            query_texts=[source_doc],
            n_results=k + 1,  # +1 because the source itself will show up
        )
        ids = (result.get("ids") or [[]])[0]
        return [i for i in ids if i != turn_id][:k]

    def delete(self, turn_id: str) -> None:
        self._coll.delete(ids=[turn_id])

    def count(self) -> int:
        return self._coll.count()
