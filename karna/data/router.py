"""Data router — the only object the agent touches for market data.

Resolution order for `get_quote(ticker)`:
    1. cache (if fresh enough — TTL depends on market hours)
    2. nsetools (lightweight live API)
    3. jugaad NSELive (alternate scraper)
    -- on success → write through to cache

Resolution order for `get_ohlcv(ticker, days)`:
    1. cache (only if enough rows are already present)
    2. jugaad stock_df (the only daily-bars source we ship by default)
    -- on success → write through to cache

Provider construction is lazy so an import failure (missing optional dep)
disables that provider rather than crashing the whole router.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from karna.data.base import DataProvider, ProviderError, Quote, is_market_open
from karna.data.cache import DataCache


log = logging.getLogger(__name__)


class DataRouter:
    """Composes providers + cache. Cheap to keep one per process."""

    def __init__(
        self,
        *,
        cache: Optional[DataCache] = None,
        quote_chain: Optional[list[str]] = None,
        ohlcv_chain: Optional[list[str]] = None,
    ):
        self.cache = cache or DataCache()
        self._providers: dict[str, DataProvider] = {}
        # yfinance is Tier 2 — slower + 15-min delayed, but works when
        # NSE's WAF blocks direct providers (which it does intermittently).
        self.quote_chain = quote_chain or ["nsetools", "jugaad", "yfinance"]
        self.ohlcv_chain = ohlcv_chain or ["jugaad", "yfinance"]

    # ---------- public ----------

    def get_quote(self, ticker: str, *, refresh: bool = False) -> Quote:
        ticker = ticker.upper()
        open_now = is_market_open()

        if not refresh:
            cached = self.cache.get_quote(ticker, market_open=open_now)
            if cached is not None:
                log.debug("quote cache hit for %s (%s)", ticker, cached.source)
                return cached

        last_err: Optional[Exception] = None
        for name in self.quote_chain:
            prov = self._get(name)
            if prov is None:
                continue
            try:
                q = prov.get_quote(ticker)
                self.cache.put_quote(q)
                return q
            except Exception as e:
                # Per-attempt failures are NOT errors — the chain exists to
                # absorb them. Default to debug so /price isn't noisy; the
                # final raise below surfaces the issue if everything fails.
                log.debug("quote provider %s failed for %s: %s", name, ticker, e)
                last_err = e
                continue
        raise ProviderError("router", f"all providers failed for {ticker}: {last_err}")

    def get_ohlcv(
        self,
        ticker: str,
        *,
        days: int = 90,
        interval: str = "1d",
        refresh: bool = False,
    ) -> pd.DataFrame:
        ticker = ticker.upper()

        if not refresh:
            cached = self.cache.get_ohlcv(ticker, interval=interval, days=days)
            if cached is not None:
                log.debug("ohlcv cache hit for %s (%d rows)", ticker, len(cached))
                return cached

        last_err: Optional[Exception] = None
        for name in self.ohlcv_chain:
            prov = self._get(name)
            if prov is None:
                continue
            try:
                df = prov.get_ohlcv(ticker, days=days, interval=interval)
                self.cache.put_ohlcv(ticker, df, interval=interval)
                return df
            except Exception as e:
                log.debug("ohlcv provider %s failed for %s: %s", name, ticker, e)
                last_err = e
                continue
        raise ProviderError("router", f"all ohlcv providers failed for {ticker}: {last_err}")

    # ---------- internals ----------

    def _get(self, name: str) -> Optional[DataProvider]:
        """Lazy-construct a provider on first use; cache the instance."""
        if name in self._providers:
            return self._providers[name]
        try:
            if name == "jugaad":
                from karna.data.jugaad import JugaadProvider
                self._providers[name] = JugaadProvider()
            elif name == "nsetools":
                from karna.data.nsetools_provider import NSEToolsProvider
                self._providers[name] = NSEToolsProvider()
            elif name == "yfinance":
                from karna.data.yfinance_provider import YFinanceProvider
                self._providers[name] = YFinanceProvider()
            else:
                log.warning("unknown provider %r in chain", name)
                self._providers[name] = None  # type: ignore[assignment]
        except Exception as e:
            log.warning("provider %s unavailable: %s", name, e)
            self._providers[name] = None  # type: ignore[assignment]
        return self._providers[name]


# Module-level singleton — agents/CLI grab it via get_router()
_default: Optional[DataRouter] = None


def get_router() -> DataRouter:
    global _default
    if _default is None:
        _default = DataRouter()
    return _default
