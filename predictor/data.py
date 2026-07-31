"""OHLC data loading for forex pairs."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

# yfinance intraday history limits (days) per interval
MAX_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "1h": 730, "1d": 3650}

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
}


def resolve_symbol(pair: str) -> str:
    key = pair.upper().replace("/", "").replace("-", "")
    return PAIRS.get(key, f"{key}=X")


def fetch_ohlc(pair: str, interval: str = "5m", days: int | None = None) -> pd.DataFrame:
    """Download OHLC candles. Returns a DataFrame indexed by UTC timestamp."""
    if interval not in MAX_DAYS:
        raise ValueError(f"unsupported interval {interval}; use one of {sorted(MAX_DAYS)}")
    period_days = min(days or MAX_DAYS[interval], MAX_DAYS[interval])
    df = yf.download(
        resolve_symbol(pair),
        period=f"{period_days}d",
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        raise RuntimeError(f"no data returned for {pair} {interval}")
    return df
