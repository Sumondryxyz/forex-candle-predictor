"""Candle sources for the batch (pandas) path used by ``train_backtest.py``.

Three backends are available:

- ``tradingview`` (default): the public TradingView chart websocket. No login,
  reachable from datacenter IPs, works out of the box. History is capped at
  ~5100 candles, so 1m only reaches back ~5.5 days.
- ``dukascopy``: free per-day 1m candle files going back years. Slower to pull
  (one request per day) but enough history to actually test 1m/5m. Research
  only; the live app does not use it.
- ``quotex``: the broker's own feed through the community ``pyquotex`` client,
  which reverse-engineers the site's WebSocket. Needs a login that Cloudflare
  blocks from datacenter IPs.

The web app uses its own streaming feeds in ``predictor/*_live.py``; this module
is only for the offline research/backtest script.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

FetchOHLC = Callable[[str, str, "int | None"], pd.DataFrame]

SOURCES = ("tradingview", "dukascopy", "quotex")
DEFAULT_SOURCE = "tradingview"


def get_fetcher(name: str | None = None) -> FetchOHLC:
    """Return the OHLC fetcher for ``name`` (default: TradingView)."""
    source = (name or DEFAULT_SOURCE).lower()
    if source == "quotex":
        from .quotex import fetch_ohlc
    elif source == "dukascopy":
        from .dukascopy import fetch_ohlc
    elif source == "tradingview":
        from .tradingview import fetch_ohlc
    else:
        raise ValueError(f"unknown source {source!r}; use one of {sorted(SOURCES)}")
    return fetch_ohlc
