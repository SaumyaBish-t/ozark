"""Thin wrapper around the local Ollama server.

We don't lean on the ollama Python SDK's higher-level helpers because we want
explicit control over messages and easy swap for the OpenAI fallback later.

The protocol Karna uses internally is OpenAI-style: a list of
{"role": "system"|"user"|"assistant", "content": str} dicts.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import httpx

from karna.config import settings


log = logging.getLogger(__name__)

Message = dict[str, str]


class OllamaError(RuntimeError):
    """Raised when the local Ollama server is unreachable or returns an error."""


class OllamaClient:
    """Talks to Ollama's /api/chat endpoint.

    One instance is meant to be reused across the agent's lifetime — keeps an
    httpx.Client around so connections to localhost are pooled.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self._client = httpx.Client(timeout=timeout)

    # ---------- public API ----------

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.4,
        model: Optional[str] = None,
    ) -> str:
        """Single-shot chat completion. Returns the assistant's text reply."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            r = self._client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama chat failed: {e}") from e

        data = r.json()
        # Ollama returns {"message": {"role": "assistant", "content": "..."}}
        return data.get("message", {}).get("content", "").strip()

    def stream_chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.4,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream tokens as they're generated. Yields content chunks."""
        import json as _json  # local import keeps top-level light

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            with self._client.stream("POST", f"{self.host}/api/chat", json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = _json.loads(line)
                    if chunk.get("done"):
                        break
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama stream failed: {e}") from e

    def embed(
        self,
        text: str | list[str],
        *,
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Embed text(s) into vectors. Returns a list of vectors (one per input).

        Uses the embedding model from settings.ollama_embed_model unless
        overridden. Caller passes either a single string (gets a 1-element
        list back) or a list of strings.
        """
        from karna.config import settings  # local import to avoid cycle on module load

        single = isinstance(text, str)
        texts = [text] if single else list(text)
        target_model = model or settings.ollama_embed_model

        # Ollama's /api/embeddings is one-prompt-at-a-time. For batches we
        # loop — Karna's embedding volume is low and serial is fine.
        out: list[list[float]] = []
        try:
            for t in texts:
                r = self._client.post(
                    f"{self.host}/api/embeddings",
                    json={"model": target_model, "prompt": t},
                )
                r.raise_for_status()
                vec = r.json().get("embedding")
                if not vec:
                    raise OllamaError(f"Ollama returned empty embedding for: {t[:60]!r}")
                out.append(vec)
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama embed failed: {e}") from e
        return out

    def close(self) -> None:
        self._client.close()


# Convenience module-level singleton for the common case.
_default: Optional[OllamaClient] = None


def get_client() -> OllamaClient:
    """Lazy singleton — callers shouldn't have to thread one through everywhere."""
    global _default
    if _default is None:
        _default = OllamaClient()
    return _default
