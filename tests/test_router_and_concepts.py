"""Pure-function tests — no infra needed."""

from __future__ import annotations

from karna.agent.core import detect_concepts
from karna.agent.router import route
from karna.trading.concepts import CONCEPT_TREE, by_category, get


def test_concept_tree_has_prereqs_resolved() -> None:
    ids = {c.id for c in CONCEPT_TREE}
    for c in CONCEPT_TREE:
        for p in c.prereqs:
            assert p in ids, f"{c.id} has unknown prereq {p!r}"


def test_get_lookup() -> None:
    assert get("rsi").name == "RSI"


def test_categories_grouped() -> None:
    cats = by_category()
    assert "indicators" in cats
    assert any(c.id == "rsi" for c in cats["indicators"])


def test_detect_concepts_basic() -> None:
    hits = detect_concepts("explain RSI please")
    assert "rsi" in hits


def test_detect_concepts_multiple() -> None:
    hits = detect_concepts("how does MACD differ from bollinger bands?")
    assert "macd" in hits
    assert "bollinger_bands" in hits


def test_router_smalltalk() -> None:
    r = route("hi there")
    assert r.intent == "smalltalk"


def test_router_question() -> None:
    r = route("what is the moving average crossover?")
    assert r.intent == "question"


def test_router_trade_action() -> None:
    r = route("paper buy RELIANCE at 2850")
    assert r.intent == "trade_action"


def test_router_system_command() -> None:
    r = route("/forget my last trade")
    assert r.intent == "system"
