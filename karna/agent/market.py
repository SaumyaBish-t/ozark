"""Agent-side glue for market data.

Two surfaces:
    1. Deterministic helpers for explicit CLI commands (/price, /rsi, …).
       These bypass the LLM entirely — fast and exact.

    2. A "market intent" pipeline that detects natural-language queries
       like "what's RSI on RELIANCE" or "show me TCS chart", fetches data,
       and hands the LLM a context block with real numbers so the reply
       is grounded in actual prices.

The router (karna/agent/router.py) gains a new intent label, "market_query",
that the agent core routes through fetch_for_message() before calling LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from karna.data.base import Quote
from karna.data.router import DataRouter, get_router as get_data_router
from karna.trading import indicators


log = logging.getLogger(__name__)


# ---------- ticker extraction ----------

# NSE tickers are uppercase, 1-15 chars, letters/digits/&. To avoid false
# positives we require either (a) explicit cue words nearby or (b) uppercase.
# Inline (?i:...) makes ONLY the cue words case-insensitive — the capture
# group stays strict uppercase, otherwise "of RELIANCE" would capture "of".
_CUE_TICKER = re.compile(
    r"\b(?i:on|for|of|in|show|chart|price|quote)\s+([A-Z][A-Z0-9&]{1,14})\b",
)
_ALL_CAPS_TICKER = re.compile(r"\b([A-Z][A-Z0-9&]{2,14})\b")

# Common English words that match the ticker regex — exclude them.
_STOP_WORDS = {
    "RSI", "MACD", "ATR", "SMA", "EMA", "BB", "BSE", "NSE", "NIFTY", "BANKNIFTY",
    "USD", "INR", "OK", "OHLC", "OHLCV", "API", "VWAP", "ETF", "IPO", "FII", "DII",
    "AND", "THE", "FOR", "WITH", "FROM", "WHAT", "WHY", "HOW", "WHEN",
}


def extract_ticker(message: str) -> Optional[str]:
    """Best-effort ticker extraction from free-form text.

    Prefers cue-word matches, falls back to uppercase tokens. Returns
    None when no plausible ticker is found.
    """
    m = _CUE_TICKER.search(message)
    if m:
        cand = m.group(1).upper()
        if cand not in _STOP_WORDS:
            return cand
    # Fall back to scanning ALL-CAPS tokens, skipping stop-words / known indicator names.
    for cand in _ALL_CAPS_TICKER.findall(message):
        cand = cand.upper()
        if cand in _STOP_WORDS:
            continue
        # Reject if it's the message's only uppercase word AND short — likely not a ticker.
        if len(cand) < 3:
            continue
        return cand
    return None


# ---------- indicator extraction ----------

INDICATOR_KEYWORDS = {
    "rsi": "rsi",
    "macd": "macd",
    "bollinger": "bb",
    "bb": "bb",
    "bands": "bb",
    "atr": "atr",
    "sma": "sma",
    "moving average": "sma",
    "ema": "ema",
}


def extract_indicators(message: str) -> list[str]:
    """Return list of indicator short-names mentioned in the message."""
    text = message.lower()
    hits: list[str] = []
    for kw, short in INDICATOR_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}\b", text) and short not in hits:
            hits.append(short)
    return hits


# ---------- detection ----------

# Heuristics for when a message is asking about live market data rather than
# a conceptual question. Used by the router as a fast pre-LLM check.
_MARKET_VERBS = re.compile(
    r"\b(price|quote|chart|level|rsi|macd|bollinger|atr|sma|ema|"
    r"moving\s+average|show|fetch|get)\b",
    re.IGNORECASE,
)


def looks_like_market_query(message: str) -> bool:
    if not _MARKET_VERBS.search(message):
        return False
    return extract_ticker(message) is not None


# ---------- handlers ----------

@dataclass
class MarketContext:
    """What we fetched in response to a message. Used to build LLM context
    and shown directly in CLI debug output."""
    ticker: str
    quote: Optional[Quote] = None
    indicator_results: dict[str, dict] = field(default_factory=dict)
    ohlcv_rows: int = 0
    errors: list[str] = field(default_factory=list)


def fetch_for_message(
    message: str,
    *,
    router: Optional[DataRouter] = None,
    days: int = 90,
) -> Optional[MarketContext]:
    """Look at a message → fetch the relevant market data → return a context
    bundle the agent can format and/or feed to the LLM.

    Returns None when no ticker can be extracted.
    """
    ticker = extract_ticker(message)
    if not ticker:
        return None

    router = router or get_data_router()
    ctx = MarketContext(ticker=ticker)

    # Quote — almost always useful, fetch first.
    try:
        ctx.quote = router.get_quote(ticker)
    except Exception as e:
        ctx.errors.append(f"quote: {e}")

    # If indicators are requested, we need OHLCV. Otherwise skip (saves the call).
    wanted = extract_indicators(message)
    if wanted:
        try:
            df = router.get_ohlcv(ticker, days=days)
            ctx.ohlcv_rows = len(df)
            for short in wanted:
                try:
                    ctx.indicator_results[short] = indicators.compute(short, df)
                except Exception as e:
                    ctx.errors.append(f"{short}: {e}")
        except Exception as e:
            ctx.errors.append(f"ohlcv: {e}")

    return ctx


# ---------- formatting ----------

def format_quote(q: Quote) -> str:
    """Plain-text formatting of a quote, suitable for Telegram + CLI."""
    arrow = "↑" if q.change >= 0 else "↓"
    return (
        f"*{q.ticker}*  ₹{q.last_price:,.2f}  {arrow} ₹{abs(q.change):.2f} "
        f"({q.change_pct:+.2f}%)\n"
        f"O ₹{q.open:,.2f}  H ₹{q.high:,.2f}  L ₹{q.low:,.2f}  PC ₹{q.prev_close:,.2f}\n"
        f"Vol {q.volume:,}  via {q.source}"
    )


def format_indicator(short: str, result: dict) -> str:
    name = short.upper()
    if result.get("current") is None:
        return f"*{name}*: insufficient data"
    return f"*{name}*: {result['interpretation']}"


def format_context_for_llm(ctx: MarketContext) -> str:
    """Inject real numbers into the system prompt so the LLM can ground its reply."""
    lines = [f"Market data fetched for {ctx.ticker}:"]
    if ctx.quote:
        q = ctx.quote
        lines.append(
            f"  last price ₹{q.last_price:.2f}, change {q.change_pct:+.2f}%, "
            f"OHLC ({q.open:.2f}/{q.high:.2f}/{q.low:.2f}/{q.prev_close:.2f}), "
            f"volume {q.volume:,}, source {q.source}"
        )
    if ctx.indicator_results:
        for short, res in ctx.indicator_results.items():
            lines.append(f"  {short.upper()}: {res.get('interpretation','n/a')}")
    if ctx.errors:
        lines.append(f"  errors: {'; '.join(ctx.errors)}")
    lines.append("Use these numbers in your explanation — don't invent values.")
    return "\n".join(lines)
