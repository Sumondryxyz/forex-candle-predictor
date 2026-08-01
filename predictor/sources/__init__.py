"""Pluggable candle sources.

``DATA_SOURCE`` selects the backend: ``yfinance`` (default) or ``quotex``.
"""
from __future__ import annotations

import os
from collections.abc import Callable

import pandas as pd

FetchOHLC = Callable[[str, str, "int | None"], pd.DataFrame]


def get_fetcher(name: str | None = None) -> FetchOHLC:
    source = (name or os.getenv("DATA_SOURCE", "yfinance")).lower()
    if source in {"yfinance", "yahoo"}:
        from ..data import fetch_ohlc

        return fetch_ohlc
    if source == "quotex":
        from .quotex import fetch_ohlc as quotex_fetch

        return quotex_fetch
    raise ValueError(f"unknown DATA_SOURCE {source!r}; use 'yfinance' or 'quotex'")
