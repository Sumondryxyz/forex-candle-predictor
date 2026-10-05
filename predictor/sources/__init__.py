"""Candle source: the Quotex broker feed.

The bot reads candles through the community ``pyquotex`` client, which
reverse-engineers the site's WebSocket. Quotex is the only backend, so
``get_fetcher()`` always returns the Quotex fetcher.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

FetchOHLC = Callable[[str, str, "int | None"], pd.DataFrame]


def get_fetcher(name: str | None = None) -> FetchOHLC:
    from .quotex import fetch_ohlc

    return fetch_ohlc
