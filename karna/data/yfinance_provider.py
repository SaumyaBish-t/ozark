"""yfinance fallback — Yahoo Finance via the unofficial Yahoo API.

Used when nsetools / jugaad get blocked by NSE's WAF. yfinance routes
through Yahoo's servers so it isn't affected by NSE rate-limits/bans.

Quirks:
    - Indian stocks need ".NS" suffix (RELIANCE → RELIANCE.NS).
    - Real-time quotes are 15-min delayed during market hours.
    - Historical data is solid; this is our most reliable OHLCV source
      when NSE is blocking.

Install note: yfinance pins beautifulsoup4>=4.11 but jugaad-data 0.27
hard-pins it to 4.9.3. pip's resolver can't reconcile, so install with:
    pip install --no-deps yfinance==0.2.43
yfinance works fine with the bs4 4.9 that jugaad brings in.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from karna.data.base import (
    DataProvider,
    OHLCV_COLS,
    ProviderError,
    Quote,
)


log = logging.getLogger(__name__)


def _suffix(ticker: str) -> str:
    """Add the .NS suffix Yahoo needs for NSE listings."""
    t = ticker.upper()
    if "." in t:
        return t
    return f"{t}.NS"


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self):
        try:
            import yfinance as yf
        except ImportError as e:
            raise ProviderError(
                self.name,
                f"yfinance not installed. Run: pip install --no-deps yfinance==0.2.43"
            ) from e
        self._yf = yf

    # ---------- live-ish quote (15-min delayed) ----------

    def get_quote(self, ticker: str) -> Quote:
        symbol = _suffix(ticker)
        try:
            tkr = self._yf.Ticker(symbol)
            info = tkr.fast_info
            hist = tkr.history(period="2d", interval="1d")
        except Exception as e:
            raise ProviderError(self.name, f"Ticker({symbol}) failed: {e}")

        if hist is None or hist.empty:
            raise ProviderError(self.name, f"no recent bars for {symbol}")

        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else last
        last_close = float(last["Close"])
        prev_close = float(prev["Close"])
        change = last_close - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        return Quote(
            ticker=ticker.upper(),
            last_price=last_close,
            change=change,
            change_pct=change_pct,
            open=float(last["Open"]),
            high=float(last["High"]),
            low=float(last["Low"]),
            prev_close=prev_close,
            volume=int(last["Volume"]),
            timestamp=time.time(),
            source=self.name,
            extras={
                "delayed_minutes": 15,
                "yahoo_symbol": symbol,
                "currency": "INR",
            },
        )

    # ---------- historical ----------

    def get_ohlcv(self, ticker: str, *, days: int = 90, interval: str = "1d") -> pd.DataFrame:
        if interval != "1d":
            raise ProviderError(self.name, f"only '1d' supported here, got {interval!r}")
        symbol = _suffix(ticker)
        # yfinance period strings: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
        period = "1y" if days <= 365 else ("2y" if days <= 730 else "max")
        try:
            df = self._yf.Ticker(symbol).history(period=period, interval=interval)
        except Exception as e:
            raise ProviderError(self.name, f"history failed for {symbol}: {e}")

        if df is None or df.empty:
            raise ProviderError(self.name, f"empty history for {symbol}")

        out = df.reset_index().rename(columns={"index": "Date"})
        # yfinance returns Date as the index name; reset_index gives a Date column.
        if "Date" not in out.columns:
            out = out.rename(columns={out.columns[0]: "Date"})
        out = out[list(OHLCV_COLS)].copy()
        out["Date"] = pd.to_datetime(out["Date"])
        return out.sort_values("Date").reset_index(drop=True).tail(days).reset_index(drop=True)
