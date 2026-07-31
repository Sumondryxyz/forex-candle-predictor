#!/usr/bin/env python3
"""Print live Quotex candles in the terminal.

Standalone diagnostic: no model, no Telegram. Use it to confirm the broker
connection works from your device (Cloudflare blocks datacenter IPs, so run it
on a phone or home connection).

    pip install "git+https://github.com/cleitonleonel/pyquotex.git"
    export QUOTEX_EMAIL=... QUOTEX_PASSWORD=...
    python quotex_stream.py --pair EURUSD --interval 1m
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from datetime import datetime, timezone

PERIODS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


def _row(candle: dict) -> str:
    direction = "UP  " if candle["close"] >= candle["open"] else "DOWN"
    return (
        f"{_stamp(candle['time'])}  O {candle['open']:.5f}  H {candle['high']:.5f}  "
        f"L {candle['low']:.5f}  C {candle['close']:.5f}  {direction}"
    )


async def run(pair: str, interval: str, history: int, live: bool) -> None:
    from pyquotex.stable_api import Quotex

    email, password = os.getenv("QUOTEX_EMAIL"), os.getenv("QUOTEX_PASSWORD")
    if not email or not password:
        raise SystemExit("set QUOTEX_EMAIL and QUOTEX_PASSWORD first")

    client = Quotex(email=email, password=password, lang="en")
    client.set_account_mode(os.getenv("QUOTEX_ACCOUNT", "PRACTICE"))

    print(f"connecting to Quotex as {email} ...")
    ok, reason = await client.connect()
    print("connected" if ok else f"login failed: {reason}")
    if not ok:
        return

    try:
        print(f"balance: {await client.get_balance()}")
        period = PERIODS[interval]
        candles = await client.get_candles(pair, time.time(), history * period, period)
        print(f"\nlast {len(candles or [])} {interval} candles of {pair}:")
        for candle in candles or []:
            print(_row(candle))

        if not live:
            return

        print("\nstreaming live candles, Ctrl+C to stop")
        await client.start_candles_stream(pair, period)
        seen = set()
        while True:
            await asyncio.sleep(1)
            for candle in client.get_realtime_candles(pair, period).values():
                key = candle.get("time")
                if key and key not in seen and candle.get("close") is not None:
                    seen.add(key)
                    print(_row(candle))
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="EURUSD", help="e.g. EURUSD or EURUSD_otc")
    ap.add_argument("--interval", default="1m", choices=sorted(PERIODS))
    ap.add_argument("--history", type=int, default=20, help="how many past candles to print")
    ap.add_argument("--no-live", action="store_true", help="print history and exit")
    args = ap.parse_args()
    asyncio.run(run(args.pair, args.interval, args.history, not args.no_live))


if __name__ == "__main__":
    main()
