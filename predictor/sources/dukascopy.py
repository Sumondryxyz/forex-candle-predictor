"""Dukascopy historical tick-candle feed (research only).

TradingView caps history at ~5100 candles, so 1m sees only ~5.5 days -- far too
little to tell whether the fast timeframes have an edge. Dukascopy publishes
free per-day 1m candle files going back years, which is enough history to test
properly.

Layout: ``{pair}/{year}/{month-1:02d}/{day:02d}/BID_candles_min_1.bi5``. Each
file is LZMA-compressed; the payload is 24-byte big-endian records of
``int32 ms_since_midnight, int32 open, close, low, high (x1e5), float32 volume``.

This module is for the offline research scripts; the live app does not use it.
"""
from __future__ import annotations

import datetime as dt
import lzma
import struct
import time
import urllib.error
import urllib.request

import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed"
_UA = "Mozilla/5.0 (X11; Linux x86_64)"
_RECORD = 24


def _fetch_day(pair: str, day: dt.date, *, retries: int = 5) -> list[tuple]:
    url = (f"{BASE}/{pair.upper()}/{day.year}/{day.month - 1:02d}/"
           f"{day.day:02d}/BID_candles_min_1.bi5")
    delay = 1.0
    raw = b""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as exc:
            # Weekends and holidays simply have no file (404).
            if exc.code == 404:
                return []
            if attempt == retries:
                return []  # a missing burst is not fatal; skip the day
            time.sleep(delay)
            delay = min(delay * 2, 20.0)
        except (urllib.error.URLError, TimeoutError, OSError):
            # A burst of requests earns a timeout/429/503; back off and retry.
            if attempt == retries:
                return []
            time.sleep(delay)
            delay = min(delay * 2, 20.0)
    if not raw:
        return []
    return _decode(raw, day)


def _decode(raw: bytes, day: dt.date) -> list[tuple]:
    """Decode one day's LZMA payload into ``(ts, o, h, l, c, v)`` rows."""
    payload = lzma.decompress(raw)
    out = []
    midnight = dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC)
    for i in range(0, len(payload) - _RECORD + 1, _RECORD):
        ms, o, c, low, high, vol = struct.unpack(">5if", payload[i:i + _RECORD])
        ts = midnight + dt.timedelta(milliseconds=ms)
        out.append((ts, o / 1e5, high / 1e5, low / 1e5, c / 1e5, vol))
    return out


def fetch_ohlc(pair: str, interval: str = "1m", days: int | None = None,
               *, end: dt.date | None = None, pause: float = 0.35) -> pd.DataFrame:
    """Return ``days`` of ``interval`` candles ending at ``end`` (default today).

    Only 1m is published at this granularity; slower intervals are resampled
    from it so a 5m/15m/1h backtest uses exactly the same underlying prices.
    """
    if interval not in ("1m", "5m", "15m", "1h"):
        raise ValueError("dukascopy feed supports 1m, 5m, 15m, 1h")
    end = end or dt.datetime.now(dt.UTC).date()
    days = days or 90
    rows: list[tuple] = []
    for back in range(days, -1, -1):
        day = end - dt.timedelta(days=back)
        rows.extend(_fetch_day(pair, day))
        time.sleep(pause)
    if not rows:
        raise RuntimeError(f"dukascopy returned no candles for {pair}")
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("time").set_index("time").sort_index()
    if interval != "1m":
        df = (df.resample({"5m": "5min", "15m": "15min", "1h": "1h"}[interval])
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"}).dropna())
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pair")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    frame = fetch_ohlc(args.pair, args.interval, args.days)
    print(frame.tail())
    print(f"{len(frame)} rows, {frame.index[0]} .. {frame.index[-1]}")
