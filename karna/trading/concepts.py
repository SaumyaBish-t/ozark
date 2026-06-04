"""Trading concept tree (seed data for Neo4j Layer 3 memory).

Each concept has:
    - id          : stable snake_case identifier used as Neo4j node key
    - name        : human-readable label
    - category    : top-level grouping (price_action, indicators, risk, etc.)
    - depth       : conceptual depth (0=foundational, higher=more advanced)
    - prereqs     : list of concept ids that should be understood first

Comprehension state (initial / reviewed / solid / mastered) lives on the user
graph as a relationship attribute, not here. Here is just the static tree.

This is loaded into Neo4j by `karna.memory.graph_store.GraphStore.seed`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Concept:
    id: str
    name: str
    category: str
    depth: int
    prereqs: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""


# Ordered so prerequisites appear before dependents (helpful for debugging,
# not required by the seed code — Cypher MERGE is order-independent).
CONCEPT_TREE: list[Concept] = [
    # ---- Foundations ----
    Concept("price_action", "Price action", "price_action", 0,
            summary="Reading raw price movement on a chart without indicators."),
    Concept("percentage_change", "Percentage change", "math", 0,
            summary="(new - old) / old. Backbone of returns and risk math."),
    Concept("std_dev", "Standard deviation", "math", 1,
            prereqs=("percentage_change",),
            summary="Spread of returns around the mean. Used in volatility metrics."),
    Concept("expected_value", "Expected value", "math", 1,
            summary="Probability-weighted average outcome. Decides if a strategy has edge."),
    Concept("correlation", "Correlation", "math", 1,
            prereqs=("std_dev",), summary="How two assets move together."),

    # ---- Candlestick patterns ----
    Concept("candlestick_anatomy", "Candle anatomy", "price_action", 1,
            prereqs=("price_action",), summary="OHLC, body vs wick, bullish vs bearish candles."),
    Concept("doji", "Doji", "candlesticks", 2, prereqs=("candlestick_anatomy",)),
    Concept("hammer", "Hammer", "candlesticks", 2, prereqs=("candlestick_anatomy",)),
    Concept("shooting_star", "Shooting star", "candlesticks", 2, prereqs=("candlestick_anatomy",)),
    Concept("engulfing", "Engulfing pattern", "candlesticks", 3, prereqs=("candlestick_anatomy",)),
    Concept("morning_star", "Morning star", "candlesticks", 3, prereqs=("candlestick_anatomy",)),

    # ---- Structure ----
    Concept("trend", "Trend identification", "structure", 1, prereqs=("price_action",)),
    Concept("support_resistance", "Support & resistance", "structure", 2, prereqs=("price_action",)),
    Concept("trendlines", "Trendlines", "structure", 2, prereqs=("support_resistance",)),
    Concept("volume_basics", "Volume basics", "structure", 1, prereqs=("price_action",)),
    Concept("volume_price_confirmation", "Volume + price confirmation", "structure", 2,
            prereqs=("volume_basics", "support_resistance")),

    # ---- Chart patterns ----
    Concept("chart_patterns", "Chart patterns", "patterns", 3,
            prereqs=("candlestick_anatomy", "support_resistance")),
    Concept("head_shoulders", "Head & shoulders", "patterns", 4, prereqs=("chart_patterns",)),
    Concept("double_top", "Double top / bottom", "patterns", 4, prereqs=("chart_patterns",)),
    Concept("flags_pennants", "Flags & pennants", "patterns", 4, prereqs=("chart_patterns", "trend")),
    Concept("triangles", "Triangles", "patterns", 4, prereqs=("chart_patterns",)),

    # ---- Indicators ----
    Concept("indicators", "Indicators overview", "indicators", 2, prereqs=("price_action",)),
    Concept("momentum", "Momentum", "indicators", 2, prereqs=("percentage_change",)),
    Concept("moving_averages", "Moving averages (SMA/EMA)", "indicators", 2,
            prereqs=("indicators",)),
    Concept("rsi", "RSI", "indicators", 3,
            prereqs=("indicators", "momentum"),
            summary="Relative Strength Index, 0-100 oscillator."),
    Concept("macd", "MACD", "indicators", 3,
            prereqs=("moving_averages", "momentum")),
    Concept("stochastic", "Stochastic oscillator", "indicators", 3,
            prereqs=("momentum",)),
    Concept("bollinger_bands", "Bollinger Bands", "indicators", 3,
            prereqs=("moving_averages", "std_dev")),
    Concept("atr", "ATR (Average True Range)", "indicators", 3, prereqs=("indicators",)),

    # ---- Risk management (parallel track) ----
    Concept("risk_per_trade", "Risk per trade", "risk", 1,
            prereqs=("percentage_change",)),
    Concept("risk_reward_ratio", "Risk:reward ratio", "risk", 2,
            prereqs=("risk_per_trade",)),
    Concept("stop_loss_types", "Stop loss types", "risk", 2,
            prereqs=("support_resistance", "atr")),
    Concept("position_sizing", "Position sizing", "risk", 3,
            prereqs=("risk_per_trade", "stop_loss_types")),
    Concept("emotional_discipline", "Emotional discipline", "risk", 1),

    # ---- Strategy ----
    Concept("entry_rules", "Entry rules", "strategy", 4,
            prereqs=("indicators", "chart_patterns")),
    Concept("exit_rules", "Exit rules", "strategy", 4,
            prereqs=("stop_loss_types", "risk_reward_ratio")),
    Concept("backtesting", "Backtesting", "strategy", 4,
            prereqs=("entry_rules", "exit_rules", "expected_value")),
    Concept("strategy", "Strategy (composite)", "strategy", 5,
            prereqs=("entry_rules", "exit_rules", "position_sizing", "backtesting")),
]


def get(concept_id: str) -> Concept:
    """Lookup by id. Raises KeyError if unknown."""
    for c in CONCEPT_TREE:
        if c.id == concept_id:
            return c
    raise KeyError(concept_id)


def by_category() -> dict[str, list[Concept]]:
    out: dict[str, list[Concept]] = {}
    for c in CONCEPT_TREE:
        out.setdefault(c.category, []).append(c)
    return out
