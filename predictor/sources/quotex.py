"""Quotex candle source (unofficial, use at your own risk).

Quotex publishes no API. This module talks to the broker through the community
``pyquotex`` client, which reverse-engineers the site's WebSocket. Using it
breaks Quotex's terms of service and can get an account banned.

Cloudflare blocks logins from datacenter IPs, so this only works when the bot
runs from a residential connection (or through a residential proxy set via
``QUOTEX_PROXY``). If a browser session is already logged in, export its cookies
to ``QUOTEX_SESSION_FILE`` (JSON with ``cookies``, ``token`` and ``user_agent``)
and no password is needed.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pandas as pd

from .. import quotex_session
from ..scheduler import interval_seconds

SESSION_FILE = os.getenv("QUOTEX_SESSION_FILE", "")
HOST = quotex_session.DEFAULT_HOST
ROOT_PATH = os.getenv("QUOTEX_ROOT_PATH", str(Path.home() / ".quotex"))


def _client():
    try:
        from pyquotex.stable_api import Quotex
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pyquotex is not installed; "
            "pip install git+https://github.com/cleitonleonel/pyquotex.git "
            "(needs Python >= 3.12)"
        ) from exc

    quotex_session.apply(HOST)
    proxy = os.getenv("QUOTEX_PROXY", "").strip()
    client = Quotex(
        email=os.getenv("QUOTEX_EMAIL", ""),
        password=os.getenv("QUOTEX_PASSWORD", ""),
        lang="en",
        host=HOST,
        root_path=ROOT_PATH,
        proxies={"http": proxy, "https": proxy} if proxy else None,
    )
    client.set_account_mode(os.getenv("QUOTEX_ACCOUNT", "PRACTICE"))
    if SESSION_FILE:
        session = quotex_session.load_session(SESSION_FILE, os.getenv("QUOTEX_EMAIL"))
        client.set_session(
            user_agent=session["user_agent"],
            cookies=session.get("cookies"),
            ssid=session.get("token"),
        )
    return client


async def _download(pair: str, interval: str, candles: int) -> list[dict]:
    client = _client()
    ok, reason = await client.connect()
    if not ok:
        raise RuntimeError(f"Quotex login failed: {reason}")
    try:
        period = interval_seconds(interval)
        rows = await client.get_candles(pair.upper(), time.time(), candles * period, period)
    finally:
        await client.close()
    if not rows:
        raise RuntimeError(f"Quotex returned no candles for {pair} {interval}")
    return rows


def fetch_ohlc(pair: str, interval: str = "1m", days: int | None = None) -> pd.DataFrame:
    """Return OHLC candles indexed by UTC timestamp, same shape the engine expects."""
    period = interval_seconds(interval)
    candles = int((days or 7) * 86400 / period)
    rows = asyncio.run(_download(pair, interval, candles))
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").rename(columns={"ticks": "volume"})
    for column in ("open", "high", "low", "close"):
        if column not in df:
            raise RuntimeError(f"Quotex candle payload is missing '{column}': {sorted(rows[0])}")
    if "volume" not in df:
        df["volume"] = 0
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df[~df.index.duplicated(keep="last")].dropna()
