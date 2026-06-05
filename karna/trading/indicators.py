"""Technical indicator calculations on OHLCV DataFrames.

Wraps pandas-ta-classic (the maintained fork of pandas-ta) so the rest of
Karna doesn't have to know which TA library we use. Each function returns
a dict with at minimum:

    { "current": float, "history": pd.Series, "interpretation": str }

The interpretation strings are deliberately plain-English so they can be
dropped into LLM context or shown to the user directly without further
processing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta_classic as ta


# ---------- shape contract ----------

def _last_finite(series: pd.Series) -> Optional[float]:
    """Return the last non-NaN value or None — indicators have warm-up NaNs."""
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _require_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("OHLCV DataFrame missing 'Close' column")
    return df["Close"].astype(float)


# ---------- indicators ----------

def rsi(df: pd.DataFrame, length: int = 14) -> dict:
    """Relative Strength Index. <30 oversold, >70 overbought."""
    close = _require_close(df)
    series = ta.rsi(close, length=length)
    current = _last_finite(series)
    if current is None:
        return {"current": None, "history": series, "interpretation": "not enough data"}

    if current >= 70:
        verdict = f"overbought ({current:.1f}) — momentum is stretched, reversal risk rising"
    elif current <= 30:
        verdict = f"oversold ({current:.1f}) — selling exhausted, watch for bounce"
    elif current >= 60:
        verdict = f"bullish ({current:.1f}) — strong upward momentum, not yet stretched"
    elif current <= 40:
        verdict = f"bearish ({current:.1f}) — downward momentum, not yet washed out"
    else:
        verdict = f"neutral ({current:.1f}) — no clear momentum bias"

    return {"current": current, "history": series, "interpretation": verdict}


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD line + signal + histogram. Histogram > 0 = bullish momentum."""
    close = _require_close(df)
    out = ta.macd(close, fast=fast, slow=slow, signal=signal)
    if out is None or out.empty:
        return {"current": None, "history": None, "interpretation": "not enough data"}

    # pandas-ta column names: MACD_{f}_{s}_{sig}, MACDh_..., MACDs_...
    macd_col = f"MACD_{fast}_{slow}_{signal}"
    hist_col = f"MACDh_{fast}_{slow}_{signal}"
    signal_col = f"MACDs_{fast}_{slow}_{signal}"

    line = _last_finite(out[macd_col])
    sig = _last_finite(out[signal_col])
    hist = _last_finite(out[hist_col])

    if line is None or sig is None or hist is None:
        return {"current": None, "history": out, "interpretation": "not enough data"}

    if hist > 0 and line > sig:
        verdict = f"bullish — MACD ({line:.2f}) above signal ({sig:.2f}), histogram positive"
    elif hist < 0 and line < sig:
        verdict = f"bearish — MACD ({line:.2f}) below signal ({sig:.2f}), histogram negative"
    else:
        verdict = f"transitioning — MACD {line:.2f}, signal {sig:.2f}, watch for cross"

    return {
        "current": {"macd": line, "signal": sig, "hist": hist},
        "history": out,
        "interpretation": verdict,
    }


def bollinger(df: pd.DataFrame, length: int = 20, std: float = 2.0) -> dict:
    """Bollinger Bands. Price near upper/lower = stretched."""
    close = _require_close(df)
    out = ta.bbands(close, length=length, std=std)
    if out is None or out.empty:
        return {"current": None, "history": None, "interpretation": "not enough data"}

    upper = _last_finite(out[f"BBU_{length}_{std}"])
    middle = _last_finite(out[f"BBM_{length}_{std}"])
    lower = _last_finite(out[f"BBL_{length}_{std}"])
    last_close = _last_finite(close)

    if None in (upper, middle, lower, last_close):
        return {"current": None, "history": out, "interpretation": "not enough data"}

    width_pct = (upper - lower) / middle * 100
    if last_close >= upper:
        verdict = f"at upper band ({upper:.2f}) — overextended; reversal or breakout"
    elif last_close <= lower:
        verdict = f"at lower band ({lower:.2f}) — oversold; bounce candidate"
    else:
        pos = (last_close - lower) / (upper - lower)
        verdict = f"mid-band, {pos*100:.0f}% of the way up (width {width_pct:.1f}%)"

    return {
        "current": {"upper": upper, "middle": middle, "lower": lower, "close": last_close},
        "history": out,
        "interpretation": verdict,
    }


def atr(df: pd.DataFrame, length: int = 14) -> dict:
    """Average True Range — volatility in absolute price units. Stop-loss sizing."""
    if not all(c in df.columns for c in ("High", "Low", "Close")):
        raise ValueError("ATR needs High/Low/Close columns")
    series = ta.atr(df["High"], df["Low"], df["Close"], length=length)
    current = _last_finite(series)
    if current is None:
        return {"current": None, "history": series, "interpretation": "not enough data"}
    return {
        "current": current,
        "history": series,
        "interpretation": (
            f"average daily range ≈ ₹{current:.2f}. For a stop-loss, "
            f"1×ATR = ₹{current:.2f}, 1.5×ATR = ₹{current*1.5:.2f}"
        ),
    }


def sma(df: pd.DataFrame, length: int = 50) -> dict:
    """Simple moving average."""
    close = _require_close(df)
    series = ta.sma(close, length=length)
    current = _last_finite(series)
    last_close = _last_finite(close)
    if current is None or last_close is None:
        return {"current": None, "history": series, "interpretation": "not enough data"}
    pos = "above" if last_close > current else "below"
    diff_pct = (last_close - current) / current * 100
    return {
        "current": current,
        "history": series,
        "interpretation": (
            f"price is {pos} {length}-day SMA (₹{current:.2f}) by {diff_pct:+.2f}%"
        ),
    }


def ema(df: pd.DataFrame, length: int = 20) -> dict:
    """Exponential moving average — weights recent prices more."""
    close = _require_close(df)
    series = ta.ema(close, length=length)
    current = _last_finite(series)
    last_close = _last_finite(close)
    if current is None or last_close is None:
        return {"current": None, "history": series, "interpretation": "not enough data"}
    pos = "above" if last_close > current else "below"
    return {
        "current": current,
        "history": series,
        "interpretation": f"price is {pos} {length}-EMA (₹{current:.2f})",
    }


# ---------- dispatcher ----------

# Single map so the agent (and CLI commands) can dispatch by short name.
INDICATORS = {
    "rsi": rsi,
    "macd": macd,
    "bb": bollinger,
    "bollinger": bollinger,
    "atr": atr,
    "sma": sma,
    "ema": ema,
}


def compute(name: str, df: pd.DataFrame, **kwargs) -> dict:
    """Look up and run an indicator by short name."""
    fn = INDICATORS.get(name.lower())
    if fn is None:
        raise KeyError(f"unknown indicator {name!r}; valid: {sorted(INDICATORS)}")
    return fn(df, **kwargs)
