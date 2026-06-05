"""Historical NSE bars via jugaad_data.

jugaad_data scrapes NSE's own data pages — no API key, no rate-limit token.
But: NSE blocks IPs that hit too aggressively. The cache layer
(`karna/data/cache.py`) is what shields you from that in practice; this
file just does one fetch when asked.

Limitations:
    - Only daily ("1d") interval — NSE's bhavcopy is daily.
    - Quote-style real-time lookup uses NSELive (live class) separately.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from karna.data.base import (
    DataProvider,
    OHLCV_COLS,
    ProviderError,
    Quote,
    is_market_open,
)


log = logging.getLogger(__name__)


class JugaadProvider(DataProvider):
    """Daily OHLCV from NSE via jugaad_data, plus live quote via NSELive."""

    name = "jugaad"

    def __init__(self):
        # Import inside the constructor so an absent dep doesn't break import-time.
        try:
            from jugaad_data.nse import stock_df, NSELive
        except ImportError as e:
            raise ProviderError(self.name, f"jugaad_data not installed: {e}")
        self._stock_df = stock_df
        self._nselive = NSELive()

    # ---------- live quote ----------

    def get_quote(self, ticker: str) -> Quote:
        if not is_market_open():
            # NSELive still answers off-hours but returns the prior close.
            # Caller can decide whether to use it; we just flag in extras.
            log.debug("market closed; returning last-known snapshot")

        try:
            raw = self._nselive.stock_quote(ticker.upper())
        except Exception as e:
            raise ProviderError(self.name, f"NSELive failed for {ticker}: {e}")

        if not raw or "priceInfo" not in raw:
            raise ProviderError(self.name, f"NSELive returned empty for {ticker}")

        p = raw["priceInfo"]
        try:
            return Quote(
                ticker=ticker.upper(),
                last_price=float(p["lastPrice"]),
                change=float(p.get("change", 0.0)),
                change_pct=float(p.get("pChange", 0.0)),
                open=float(p["open"]),
                high=float(p.get("intraDayHighLow", {}).get("max", p.get("dayHigh", 0.0))),
                low=float(p.get("intraDayHighLow", {}).get("min", p.get("dayLow", 0.0))),
                prev_close=float(p["previousClose"]),
                volume=int(raw.get("preOpenMarket", {}).get("totalTradedVolume") or 0),
                timestamp=time.time(),
                source=self.name,
                extras={"vwap": p.get("vwap"), "currency": "INR"},
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(self.name, f"unexpected NSELive shape for {ticker}: {e}")

    # ---------- historical ----------

    def get_ohlcv(
        self,
        ticker: str,
        *,
        days: int = 90,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval != "1d":
            raise ProviderError(self.name, f"jugaad_data only supports interval='1d', got {interval!r}")

        to_date = date.today()
        from_date = to_date - timedelta(days=days * 2)  # cushion for weekends/holidays

        try:
            df = self._stock_df(
                symbol=ticker.upper(),
                from_date=from_date,
                to_date=to_date,
                series="EQ",
            )
        except Exception as e:
            raise ProviderError(self.name, f"stock_df failed for {ticker}: {e}")

        if df is None or df.empty:
            raise ProviderError(self.name, f"no historical rows for {ticker}")

        # jugaad columns: DATE, OPEN, HIGH, LOW, CLOSE, VOLUME, ... — normalize.
        df = df.rename(columns={
            "DATE": "Date", "OPEN": "Open", "HIGH": "High",
            "LOW": "Low", "CLOSE": "Close", "VOLUME": "Volume",
        })
        out = df[list(OHLCV_COLS)].copy()
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.sort_values("Date").reset_index(drop=True)
        return out.tail(days).reset_index(drop=True)
