"""Market data provider interface.

Every concrete source (jugaad_data, nsetools, yfinance, broker APIs) implements
this. The data router (`karna/data/router.py`) is what the rest of Karna talks
to — it tries providers in order, falls back on failure, and caches successes.

We don't expose pandas DataFrames as the public contract for quotes (those
are scalar latest-prices) but we do for OHLCV (it's the natural shape for
historical data and what pandas-ta-classic expects as input).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Optional

import pandas as pd


# Indian Standard Time, hardcoded because NSE doesn't care about your tz.
IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


@dataclass
class Quote:
    """Snapshot of one ticker at one moment."""
    ticker: str
    last_price: float
    change: float          # absolute Rs change vs previous close
    change_pct: float      # percentage change vs previous close
    open: float
    high: float
    low: float
    prev_close: float
    volume: int
    timestamp: float       # unix epoch seconds
    source: str            # which provider returned this
    extras: dict = field(default_factory=dict)


@dataclass
class ProviderError(Exception):
    """Raised when a provider can't satisfy a request. Router catches and
    moves on to the next provider in the chain."""
    provider: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.provider}] {self.detail}"


# Standard OHLCV column names. Concrete providers MUST return DataFrames
# with exactly these columns (case-sensitive) so downstream code (indicators,
# cache, agent) doesn't care which provider produced the frame.
OHLCV_COLS = ("Date", "Open", "High", "Low", "Close", "Volume")


def is_market_open(now: Optional[datetime] = None) -> bool:
    """True if NSE cash market is currently open. Used to decide whether
    to even attempt a live-quote provider call. Saturday/Sunday → False."""
    now = (now or datetime.now(IST)).astimezone(IST)
    if now.weekday() >= 5:  # Sat=5, Sun=6
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


class DataProvider(ABC):
    """Abstract base for any market-data source."""

    name: str = "abstract"   # short string used in source attribution + logs

    @abstractmethod
    def get_quote(self, ticker: str) -> Quote:
        """Latest scalar price + day metrics. Raise ProviderError if you can't."""

    @abstractmethod
    def get_ohlcv(
        self,
        ticker: str,
        *,
        days: int = 90,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Historical bars. Returns a DataFrame with columns OHLCV_COLS.

        `interval` may be "1d", "1h", "15m" etc; providers should raise
        ProviderError if they don't support the requested granularity.
        """

    # ---- optional ----

    def supports_interval(self, interval: str) -> bool:
        return interval == "1d"

    def close(self) -> None:
        """Override if the provider holds resources (HTTP clients, etc)."""
        pass
