"""Candle sources for the batch (pandas) path used by ``train_backtest.py``.

Two backends are available:

- ``tradingview`` (default): the public TradingView chart websocket. No login,
  reachable from datacenter IPs, works out of the box.
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

SOURCES = ("tradingview", "quotex")
DEFAULT_SOURCE = "tradingview"


def get_fetcher(name: str | None = None) -> FetchOHLC:
    """Return the OHLC fetcher for ``name`` (default: TradingView)."""
    source = (name or DEFAULT_SOURCE).lower()
    if source == "quotex":
        from .quotex import fetch_ohlc
    elif source == "tradingview":
        from .tradingview import fetch_ohlc
    else:
        raise ValueError(f"unknown source {source!r}; use one of {sorted(SOURCES)}")
    return fetch_ohlc
