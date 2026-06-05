"""Sprint 3 tests — extraction logic, cache I/O, indicator math.

No network involved. Generates synthetic OHLCV so indicator tests are
deterministic and CI-safe.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from karna.agent.market import extract_indicators, extract_ticker, looks_like_market_query
from karna.data.base import OHLCV_COLS, Quote, is_market_open
from karna.data.cache import DataCache
from karna.trading import indicators


# ---------- ticker / intent extraction ----------

def test_extract_ticker_with_cue() -> None:
    assert extract_ticker("what is the price of RELIANCE") == "RELIANCE"
    assert extract_ticker("show me TCS chart") == "TCS"


def test_extract_ticker_uppercase_fallback() -> None:
    assert extract_ticker("HDFCBANK is consolidating") == "HDFCBANK"


def test_extract_ticker_rejects_stop_words() -> None:
    # "RSI" and "NSE" alone should not be treated as tickers
    assert extract_ticker("what is RSI") is None
    assert extract_ticker("how does NSE work") is None


def test_extract_indicators() -> None:
    hits = extract_indicators("show me RSI and macd on RELIANCE")
    assert "rsi" in hits
    assert "macd" in hits


def test_looks_like_market_query() -> None:
    assert looks_like_market_query("what is the price of RELIANCE")
    assert looks_like_market_query("show RSI on TCS")
    assert not looks_like_market_query("what is RSI")  # no ticker
    assert not looks_like_market_query("hi there")     # no market verb


# ---------- cache ----------

@pytest.fixture
def cache(tmp_path: Path) -> DataCache:
    c = DataCache(db_path=tmp_path / "cache.db")
    yield c
    c.close()


def test_quote_put_and_get(cache: DataCache) -> None:
    q = Quote(
        ticker="RELIANCE", last_price=2850.5, change=12.0, change_pct=0.42,
        open=2840.0, high=2860.0, low=2835.0, prev_close=2838.5,
        volume=1_234_567, timestamp=time.time(), source="test",
    )
    cache.put_quote(q)
    out = cache.get_quote("RELIANCE", market_open=True)
    assert out is not None
    assert out.ticker == "RELIANCE"
    assert out.last_price == pytest.approx(2850.5)


def test_quote_ttl_expires(cache: DataCache) -> None:
    q = Quote(
        ticker="TCS", last_price=4100.0, change=0.0, change_pct=0.0,
        open=4100.0, high=4100.0, low=4100.0, prev_close=4100.0,
        volume=0, timestamp=time.time() - 10_000, source="test",
    )
    cache.put_quote(q)
    # 1-second max_age — well past the stamped timestamp.
    assert cache.get_quote("TCS", max_age=1) is None


def test_ohlcv_put_then_get_enough_rows(cache: DataCache) -> None:
    dates = [date.today() - timedelta(days=i) for i in range(60, 0, -1)]
    df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open":  np.linspace(100, 200, 60),
        "High":  np.linspace(101, 201, 60),
        "Low":   np.linspace(99, 199, 60),
        "Close": np.linspace(100.5, 200.5, 60),
        "Volume": np.full(60, 10_000),
    })
    n = cache.put_ohlcv("FAKE", df)
    assert n == 60
    out = cache.get_ohlcv("FAKE", days=30)
    assert out is not None
    assert len(out) == 30
    assert list(out.columns) == list(OHLCV_COLS)


def test_ohlcv_get_returns_none_when_undersupplied(cache: DataCache) -> None:
    # Only 5 rows cached, asking for 30 → None (forces a refresh).
    dates = [date.today() - timedelta(days=i) for i in range(5)]
    df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open": [100.0] * 5, "High": [101.0] * 5,
        "Low": [99.0] * 5, "Close": [100.5] * 5, "Volume": [1_000] * 5,
    })
    cache.put_ohlcv("THIN", df)
    assert cache.get_ohlcv("THIN", days=30) is None


# ---------- indicators ----------

def _synthetic_ohlcv(n: int = 100, trend: str = "up") -> pd.DataFrame:
    """Generate a smooth uptrend/downtrend/sideways series so indicators
    produce predictable readings."""
    rng = np.arange(n, dtype=float)
    if trend == "up":
        close = 100 + rng * 0.5 + np.sin(rng / 5) * 2
    elif trend == "down":
        close = 200 - rng * 0.5 + np.sin(rng / 5) * 2
    else:
        close = 150 + np.sin(rng / 5) * 5

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low":  close - 1.0,
        "Close": close,
        "Volume": np.full(n, 10_000),
    })
    return df


def test_rsi_on_uptrend_is_above_50() -> None:
    df = _synthetic_ohlcv(trend="up")
    res = indicators.rsi(df)
    assert res["current"] is not None
    assert res["current"] > 50


def test_rsi_on_downtrend_is_below_50() -> None:
    df = _synthetic_ohlcv(trend="down")
    res = indicators.rsi(df)
    assert res["current"] is not None
    assert res["current"] < 50


def test_macd_returns_three_lines() -> None:
    df = _synthetic_ohlcv(trend="up")
    res = indicators.macd(df)
    assert res["current"] is not None
    cur = res["current"]
    assert "macd" in cur and "signal" in cur and "hist" in cur


def test_bollinger_position() -> None:
    df = _synthetic_ohlcv(trend="sideways")
    res = indicators.bollinger(df)
    assert res["current"] is not None
    upper = res["current"]["upper"]
    lower = res["current"]["lower"]
    close = res["current"]["close"]
    assert lower <= close <= upper or abs(close - upper) < 5 or abs(close - lower) < 5


def test_compute_dispatch_unknown() -> None:
    with pytest.raises(KeyError):
        indicators.compute("not_an_indicator", _synthetic_ohlcv())


# ---------- market hours ----------

def test_is_market_open_weekend() -> None:
    # Saturday at 10am IST
    from datetime import datetime
    sat = datetime(2025, 1, 4, 10, 0)
    assert not is_market_open(sat.replace(tzinfo=None) if sat.tzinfo is None else sat)
