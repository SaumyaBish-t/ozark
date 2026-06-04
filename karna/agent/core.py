"""Main agent loop.

Sprint 1 flow:
    incoming message
      → Router classifies intent
      → Load recent session history from SQLite
      → Concept detection: find any concept ids the user mentions
      → If they mentioned a concept, look up missing prereqs in Neo4j and
        nudge the system prompt to teach prereqs first
      → Call Ollama with system prompt + history + new turn
      → Store user turn + assistant reply
      → Return reply text

Later sprints layer in:
    - ChromaDB semantic recall (Sprint 2)
    - Forgetting / retention updates (Sprint 2)
    - Trade entity extraction (Sprint 4)
    - Skill execution + reflection (Sprint 6)
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from karna.agent.router import route, RoutingResult
from karna.llm.ollama import OllamaClient, get_client as get_ollama
from karna.llm.prompts import DEFAULT_TUTOR
from karna.memory.session_store import SessionStore
from karna.trading.concepts import CONCEPT_TREE


log = logging.getLogger(__name__)


# Precompute concept lookup for cheap mention detection.
# Match on either the id (snake_case) or the name (lowercase).
_CONCEPT_TERMS: dict[str, str] = {}
for _c in CONCEPT_TREE:
    _CONCEPT_TERMS[_c.id.replace("_", " ")] = _c.id
    _CONCEPT_TERMS[_c.name.lower()] = _c.id


def detect_concepts(message: str) -> list[str]:
    """Return ids of concepts mentioned in the message. Whole-word match."""
    text = message.lower()
    hits: list[str] = []
    for term, cid in _CONCEPT_TERMS.items():
        # Word boundary at start/end so "ema" doesn't match "ema" inside another word.
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text):
            if cid not in hits:
                hits.append(cid)
    return hits


@dataclass
class AgentReply:
    text: str
    intent: str
    matched_rule: str
    mentioned_concepts: list[str] = field(default_factory=list)
    prereq_focus: list[str] = field(default_factory=list)


class Agent:
    """The thing every connector (CLI, Telegram, future API) talks to.

    GraphStore is optional so the agent still runs (in a degraded way)
    when Neo4j isn't up. That's deliberate — keeps the CLI usable for
    rapid LLM-only iteration.
    """

    def __init__(
        self,
        *,
        llm: Optional[OllamaClient] = None,
        session_store: Optional[SessionStore] = None,
        graph_store=None,  # karna.memory.graph_store.GraphStore | None
        history_window: int = 12,
    ):
        self.llm = llm or get_ollama()
        self.session_store = session_store or SessionStore()
        self.graph_store = graph_store
        self.history_window = history_window

    # ---------- public entry ----------

    def handle(
        self,
        message: str,
        *,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> AgentReply:
        """Process one user message end-to-end."""
        routing = route(message, llm=self.llm)

        # Persist the user turn first so it shows up in history for future calls
        # even if generation fails mid-way.
        self.session_store.store(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=message,
            metadata_json=f'{{"intent": "{routing.intent}"}}',
        )

        mentioned = detect_concepts(message)
        prereq_focus = self._collect_prereqs(user_id, mentioned)

        system_prompt = self._build_system_prompt(routing, mentioned, prereq_focus)
        history = [t.to_chat_message() for t in self.session_store.history(session_id, self.history_window)]

        messages = [{"role": "system", "content": system_prompt}, *history]

        reply_text = self.llm.chat(messages, temperature=0.4)

        self.session_store.store(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=reply_text,
        )

        return AgentReply(
            text=reply_text,
            intent=routing.intent,
            matched_rule=routing.matched_rule,
            mentioned_concepts=mentioned,
            prereq_focus=prereq_focus,
        )

    # ---------- internals ----------

    def _collect_prereqs(self, user_id: Optional[str], mentioned: list[str]) -> list[str]:
        """Ask the graph: of the concepts the user mentioned, which prereqs
        haven't they reached 'solid' on yet? Sprint 1 = no graph → empty list."""
        if not self.graph_store or not user_id or not mentioned:
            return []
        out: list[str] = []
        for cid in mentioned:
            try:
                for p in self.graph_store.missing_prereqs(user_id, cid):
                    if p not in out:
                        out.append(p)
            except Exception as e:
                log.debug("graph prereq lookup failed for %s: %s", cid, e)
        return out

    def _build_system_prompt(
        self,
        routing: RoutingResult,
        mentioned: list[str],
        prereq_focus: list[str],
    ) -> str:
        parts = [DEFAULT_TUTOR]

        if routing.intent == "smalltalk":
            parts.append("This message is casual conversation. Keep your reply short and warm.")
        elif routing.intent == "trade_action":
            parts.append(
                "This message looks like a trade log. For Sprint 1, just acknowledge "
                "you registered the intent — full trade tracking comes in Sprint 4."
            )
        elif routing.intent == "system":
            parts.append("This is a system/meta command. Respond plainly about what you can and can't do yet.")

        if mentioned:
            parts.append(f"Concepts the user touched on: {', '.join(mentioned)}.")
        if prereq_focus:
            top = prereq_focus[:3]
            parts.append(
                "Before explaining the user's target concept, make sure they're solid on: "
                f"{', '.join(top)}. If they aren't, teach the foundational one first."
            )

        return "\n\n".join(parts)


# ---------- session id helper ----------

def new_session_id() -> str:
    return f"sess-{uuid.uuid4().hex[:12]}"
