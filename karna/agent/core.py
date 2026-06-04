"""Main agent loop.

Sprint 1+2 flow:
    incoming message
      → Router classifies intent
      → Memory facade dual-writes the user turn (SQLite + ChromaDB)
      → Load recent session history (chat continuity)
      → Concept detection: find any concept ids the user mentions
      → If they mentioned a concept, look up missing prereqs in Neo4j
      → Semantic recall: pull past turns relevant to the new message from
        anywhere in user's history, not just the current session
      → On each retrieval hit, the consolidation engine boosts neighbors
      → Build system prompt with prereq guidance + recalled context
      → Call LLM
      → Store assistant reply
      → Return reply text

Later sprints layer in:
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
from karna.memory.memory import Memory, Recall, build_memory
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
    recalled: list[Recall] = field(default_factory=list)


class Agent:
    """The thing every connector (CLI, Telegram, future API) talks to.

    Memory and GraphStore are optional so the agent still runs (degraded)
    when ChromaDB or Neo4j aren't up — keeps the CLI usable for iteration.
    """

    def __init__(
        self,
        *,
        llm: Optional[OllamaClient] = None,
        memory: Optional[Memory] = None,
        graph_store=None,  # karna.memory.graph_store.GraphStore | None
        history_window: int = 12,
        recall_k: int = 4,
    ):
        self.llm = llm or get_ollama()
        self.memory = memory or build_memory()
        self.graph_store = graph_store
        self.history_window = history_window
        self.recall_k = recall_k

        # Lazy-imported consolidator (Sprint 2). Only used if semantic store is up.
        self._consolidator = None
        if self.memory and self.memory.semantic is not None:
            from karna.memory.consolidation import Consolidator
            self._consolidator = Consolidator(self.memory)

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

        # Persist the user turn first so it shows up for future calls even
        # if generation fails mid-way. Memory.store dual-writes to L1 + L2.
        self.memory.store(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=message,
        )

        mentioned = detect_concepts(message)
        prereq_focus = self._collect_prereqs(user_id, mentioned)

        # Semantic recall — past turns from anywhere relevant to this message.
        # Skip for smalltalk (don't waste embedding calls on "hi").
        recalled: list[Recall] = []
        if routing.intent != "smalltalk":
            try:
                recalled = self.memory.recall(message, k=self.recall_k, user_id=user_id)
                # Exclude turns from this session (we already have those via history).
                recalled = [r for r in recalled if r.session_id != session_id]
            except Exception as e:
                log.debug("recall failed: %s", e)
                recalled = []

        # Consolidation: rehearsing a memory strengthens its neighbors.
        if recalled and self._consolidator is not None:
            try:
                self._consolidator.boost_neighbors([r.turn_id for r in recalled])
            except Exception as e:
                log.debug("consolidation boost failed: %s", e)

        system_prompt = self._build_system_prompt(routing, mentioned, prereq_focus, recalled)
        history = [t.to_chat_message() for t in self.memory.history(session_id, self.history_window)]

        messages = [{"role": "system", "content": system_prompt}, *history]
        reply_text = self.llm.chat(messages, temperature=0.4)

        self.memory.store(
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
            recalled=recalled,
        )

    # ---------- internals ----------

    def _collect_prereqs(self, user_id: Optional[str], mentioned: list[str]) -> list[str]:
        """Ask the graph: of the concepts the user mentioned, which prereqs
        haven't they reached 'solid' on yet?"""
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
        recalled: list[Recall],
    ) -> str:
        parts = [DEFAULT_TUTOR]

        if routing.intent == "smalltalk":
            parts.append("This message is casual conversation. Keep your reply short and warm.")
        elif routing.intent == "trade_action":
            parts.append(
                "This message looks like a trade log. For Sprint 1-2, just acknowledge "
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
        if recalled:
            bullets = []
            for r in recalled[:4]:
                snippet = r.content.replace("\n", " ")
                if len(snippet) > 220:
                    snippet = snippet[:217] + "…"
                bullets.append(f"- ({r.role}, {r.source}) {snippet}")
            parts.append(
                "Relevant context from earlier conversations (use only if useful, don't "
                "force a reference):\n" + "\n".join(bullets)
            )

        return "\n\n".join(parts)


# ---------- session id helper ----------

def new_session_id() -> str:
    return f"sess-{uuid.uuid4().hex[:12]}"
