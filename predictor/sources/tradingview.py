#!/usr/bin/env python3
"""Live candle source backed by TradingView's public chart websocket.

TradingView is reachable from anywhere (no Cloudflare challenge), so this is the
default feed for the web app and ``train_backtest.py``. It speaks the same
``~m~``-framed socket.io protocol the site's chart uses, with the public
``unauthorized_user_token`` — no login and no account required.

This is an unofficial, undocumented endpoint. It can change without notice and
using it is outside TradingView's terms. It only reads candles; the bot never
places a trade.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import string
import threading
import time

from ..scheduler import interval_seconds

WS_URL = "wss://data.tradingview.com/socket.io/websocket?from=chart%2F&type=chart"
ORIGIN = "https://data.tradingview.com"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
RESOLUTION = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60"}
DEFAULT_EXCHANGE = os.getenv("TRADINGVIEW_EXCHANGE", "FX")
MAX_BARS = 5000
CHUNK = 300

# TradingView rejects a burst of websocket handshakes with HTTP 429. Each feed
# opens a socket per poll, so every extra pair multiplies the rate and enough
# pairs push it over the limit. Pace handshakes process-wide: a minimum gap
# between opens, plus a retry/backoff that honours the 429 before giving up.
_CONNECT_LOCK = threading.Lock()
_LAST_CONNECT = 0.0
_MIN_CONNECT_GAP = float(os.getenv("TV_CONNECT_GAP", "0.6"))
_RATE_LIMIT_RETRIES = int(os.getenv("TV_RATE_RETRIES", "6"))


def _pace_connect() -> None:
    """Serialise handshakes with a small gap so a burst cannot trip the limiter."""
    global _LAST_CONNECT
    with _CONNECT_LOCK:
        wait = _LAST_CONNECT + _MIN_CONNECT_GAP - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST_CONNECT = time.monotonic()


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "Too Many Requests" in text or "InvalidStatus" in text


def symbol_for(pair: str) -> str:
    """Map a pair to a TradingView symbol, e.g. EURUSD -> FX:EURUSD."""
    pair = pair.upper().strip()
    if ":" in pair:
        return pair
    # TradingView lists no OTC pairs
    pair = pair.removesuffix("_OTC")
    return f"{DEFAULT_EXCHANGE}:{pair}"


def resolution_for(interval: str) -> str:
    if interval not in RESOLUTION:
        raise ValueError(f"unsupported interval {interval!r}; use one of {sorted(RESOLUTION)}")
    return RESOLUTION[interval]


def _frame(func: str, args: list) -> str:
    payload = json.dumps({"m": func, "p": args})
    return f"~m~{len(payload)}~m~{payload}"


def _parse(raw: str):
    """Yield ('hb', token) heartbeats and ('json', obj) messages from a frame batch."""
    for part in re.split(r"~m~\d+~m~", raw):
        if not part:
            continue
        if part.startswith("~h~"):
            yield "hb", part[3:]
            continue
        try:
            yield "json", json.loads(part)
        except json.JSONDecodeError:
            continue


class TradingViewClient:
    """Minimal client with the same async surface the feed expects."""

    def __init__(self, host: str | None = None, exchange: str | None = None):
        self.exchange = exchange or DEFAULT_EXCHANGE
        self._url = WS_URL

    async def connect(self) -> tuple[bool, str]:
        """Check that the websocket endpoint answers."""
        import websockets

        try:
            async with websockets.connect(
                self._url, additional_headers={"Origin": ORIGIN, "User-Agent": USER_AGENT}
            ) as ws:
                await ws.send(_frame("set_auth_token", ["unauthorized_user_token"]))
                await asyncio.wait_for(ws.recv(), timeout=15)
            return True, "connected"
        except Exception as exc:  # noqa: BLE001 - surfaced to the feed as the error
            return False, f"{type(exc).__name__}: {exc}"

    async def close(self) -> None:
        return None

    async def get_candles(self, pair: str, end: float, count: int, period: int) -> list[dict]:
        """Return up to ``count`` most recent closed candles, oldest first.

        Retries the handshake when TradingView answers 429, backing off so a
        burst of feeds settles instead of failing permanently.
        """
        delay = 1.0
        last_exc: Exception | None = None
        for _ in range(_RATE_LIMIT_RETRIES + 1):
            _pace_connect()
            try:
                return await self._get_candles_once(pair, end, count, period)
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limited(exc):
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20.0)
        raise last_exc if last_exc else RuntimeError("TradingView request failed")

    async def _get_candles_once(self, pair: str, end: float, count: int, period: int) -> list[dict]:
        import websockets

        interval = _interval_from_period(period)
        symbol = symbol_for(pair) if self.exchange == DEFAULT_EXCHANGE else f"{self.exchange}:{pair.upper()}"
        resolution = resolution_for(interval)
        want = max(CHUNK, min(int(count), MAX_BARS))

        bars: dict[int, list] = {}
        cs = "cs_" + "".join(random.choice(string.ascii_letters) for _ in range(12))

        async with websockets.connect(
            self._url, additional_headers={"Origin": ORIGIN, "User-Agent": USER_AGENT}
        ) as ws:
            await ws.send(_frame("set_auth_token", ["unauthorized_user_token"]))
            await ws.send(_frame("chart_create_session", [cs, ""]))
            await ws.send(
                _frame("resolve_symbol", [cs, "sds_sym_1", f'={{"adjustment":"splits","symbol":"{symbol}"}}'])
            )
            await ws.send(_frame("create_series", [cs, "sds_1", "s1", "sds_sym_1", resolution, CHUNK, ""]))

            more = 0
            max_more = max(0, (want - CHUNK + CHUNK - 1) // CHUNK)
            while len(bars) < want and more <= max_more:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=15)
                except TimeoutError:
                    break
                for kind, obj in _parse(raw):
                    if kind == "hb":
                        token = f"~h~{obj}"
                        await ws.send(f"~m~{len(token)}~m~{token}")
                        continue
                    method, params = obj.get("m"), obj.get("p")
                    if method in {"timescale_update", "du"} and params:
                        for data in params[1].values():
                            for point in data.get("s", []):
                                values = point.get("v", [])
                                if len(values) >= 5:
                                    bars[int(values[0])] = values[:6]
                    elif method == "series_completed" and len(bars) < want and more < max_more:
                        more += 1
                        await ws.send(_frame("request_more_data", [cs, "sds_1", CHUNK]))
                    elif method == "series_error":
                        raise RuntimeError(f"TradingView series error for {symbol}: {params}")

        ordered = [bars[t] for t in sorted(bars)]
        return [
            {
                "time": int(v[0]),
                "open": v[1],
                "high": v[2],
                "low": v[3],
                "close": v[4],
                "volume": float(v[5]) if len(v) > 5 and v[5] is not None else 0.0,
            }
            for v in ordered
            if v[4] is not None
        ]


def _interval_from_period(period: int) -> str:
    for interval in RESOLUTION:
        if interval_seconds(interval) == period:
            return interval
    raise ValueError(f"no TradingView resolution for {period}s candles")


def fetch_ohlc(pair: str, interval: str = "1m", days: int | None = None):
    """Return OHLC candles indexed by UTC timestamp, same shape the engine expects."""
    import pandas as pd

    period = interval_seconds(interval)
    candles = int((days or 7) * 86400 / period)
    client = TradingViewClient()
    rows = asyncio.run(client.get_candles(pair, 0.0, candles, period))
    if not rows:
        raise RuntimeError(f"TradingView returned no candles for {pair} {interval}")
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    for column in ("open", "high", "low", "close"):
        if column not in df:
            raise RuntimeError(f"TradingView candle payload is missing '{column}'")
    if "volume" not in df:
        df["volume"] = 0
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df[~df.index.duplicated(keep="last")].dropna()
