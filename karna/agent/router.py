"""Intent router — first thing every incoming message hits.

Sprint 1 implementation: cheap keyword heuristics with an LLM fallback only
when nothing matches. We don't want to spend a model call on "hi" or
obvious paper-trade commands.

Later (Sprint 6+), we'll version this as a skill with quality metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from karna.llm.ollama import OllamaClient
from karna.llm.prompts import ROUTER_CLASSIFY


Intent = Literal["question", "task", "trade_action", "system", "smalltalk"]


@dataclass
class RoutingResult:
    intent: Intent
    confidence: float       # 0..1
    matched_rule: str       # which heuristic fired, or "llm" / "default"


# Order matters — first match wins.
_TRADE_RE = re.compile(
    r"\b(paper\s+(buy|sell|short|cover)|"
    r"(buy|sell|short|cover)\s+[A-Z]{2,10}|"
    r"(target|stop\s*loss|sl)\s+(at\s+)?\d+)",
    re.IGNORECASE,
)
_SYSTEM_RE = re.compile(
    r"^\s*/?(forget|remember|pin|unpin|status|skills|quiz me|show|help)\b",
    re.IGNORECASE,
)
_SMALLTALK_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|good\s+(morning|night|evening))\b",
    re.IGNORECASE,
)
_QUESTION_HINTS = re.compile(
    r"(\?$|^(what|why|how|when|where|which|explain|teach|tell me))\b",
    re.IGNORECASE,
)


def route(message: str, llm: OllamaClient | None = None) -> RoutingResult:
    """Classify intent. `llm` is optional — only used if heuristics miss."""
    msg = message.strip()

    if _TRADE_RE.search(msg):
        return RoutingResult("trade_action", 0.9, "regex:trade")
    if _SYSTEM_RE.match(msg):
        return RoutingResult("system", 0.95, "regex:system")
    if _SMALLTALK_RE.match(msg) and len(msg) < 40:
        return RoutingResult("smalltalk", 0.85, "regex:smalltalk")
    if _QUESTION_HINTS.search(msg):
        return RoutingResult("question", 0.75, "regex:question")

    # Nothing obvious — ask the LLM (cheap, single token reply).
    if llm is not None:
        try:
            label = llm.chat(
                [{"role": "system", "content": ROUTER_CLASSIFY},
                 {"role": "user", "content": msg}],
                temperature=0.0,
            ).lower().strip().split()[0].rstrip(".,:")
            if label in ("question", "task", "trade_action", "system", "smalltalk"):
                return RoutingResult(label, 0.6, "llm")  # type: ignore[arg-type]
        except Exception:
            pass

    return RoutingResult("question", 0.4, "default")
