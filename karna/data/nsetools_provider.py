"""Live NSE quotes via the nsetools library.

Complements jugaad_data — nsetools is simpler for the get_quote path but
has no historical bar API. We let the router pick this for live snapshots
and jugaad for OHLCV.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from karna.data.base import DataProvider, ProviderError, Quote, is_market_open


log = logging.getLogger(__name__)


class NSEToolsProvider(DataProvider):
    name = "nsetools"

    def __init__(self):
        try:
            from nsetools import Nse
        except ImportError as e:
            raise ProviderError(self.name, f"nsetools not installed: {e}")
        self._nse = Nse()

    def get_quote(self, ticker: str) -> Quote:
        try:
            q = self._nse.get_quote(ticker.upper())
        except Exception as e:
            raise ProviderError(self.name, f"get_quote failed for {ticker}: {e}")
        if not q:
            raise ProviderError(self.name, f"empty quote for {ticker}")

        try:
            return Quote(
                ticker=ticker.upper(),
                last_price=float(q.get("lastPrice", q.get("last_price", 0.0))),
                change=float(q.get("change", 0.0)),
                change_pct=float(q.get("pChange", q.get("p_change", 0.0))),
                open=float(q.get("open", 0.0)),
                high=float(q.get("dayHigh", q.get("day_high", 0.0))),
                low=float(q.get("dayLow", q.get("day_low", 0.0))),
                prev_close=float(q.get("previousClose", q.get("previous_close", 0.0))),
                volume=int(q.get("totalTradedVolume", q.get("total_traded_volume", 0))),
                timestamp=time.time(),
                source=self.name,
                extras={
                    "vwap": q.get("averagePrice"),
                    "year_high": q.get("high52", q.get("year_high")),
                    "year_low": q.get("low52", q.get("year_low")),
                    "currency": "INR",
                },
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(self.name, f"unexpected nsetools shape for {ticker}: {e}")

    def get_ohlcv(self, ticker: str, *, days: int = 90, interval: str = "1d") -> pd.DataFrame:
        raise ProviderError(self.name, "nsetools has no historical bars; use jugaad")
