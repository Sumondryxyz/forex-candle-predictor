#!/usr/bin/env python3
"""Live Quotex candles for market-qx.trade accounts (working version).

Quotex sets laravel_session twice during login (guest, then authenticated) and
pyquotex forwards both, so the server treats every request as a guest. Keeping
only the last value of each cookie is what makes the websocket accept the
session.
"""
import asyncio
import os
import time
from datetime import UTC, datetime

from pyquotex.stable_api import Quotex

from predictor import quotex_session

HOST = os.getenv("QUOTEX_HOST", "market-qx.trade")
PAIR = os.getenv("PAIR", "EURUSD")
PERIOD = int(os.getenv("PERIOD", "60"))




def show(c):
    t = datetime.fromtimestamp(c["time"], tz=UTC).strftime("%H:%M:%S")
    d = "UP" if c["close"] >= c["open"] else "DOWN"
    print(
        f'{t}  O {c["open"]:.5f}  H {c["high"]:.5f}  '
        f'L {c["low"]:.5f}  C {c["close"]:.5f}  {d}'
    )


async def main():
    quotex_session.apply(HOST)
    q = Quotex(
        email=os.environ["QUOTEX_EMAIL"],
        password=os.environ["QUOTEX_PASSWORD"],
        lang="en",
        host=HOST,
    )
    q.set_account_mode("PRACTICE")

    ok, reason = await q.connect()
    print("connected" if ok else f"failed: {reason}")
    if not ok:
        return

    print(f"balance: {await q.get_balance()}   asset: {PAIR}   period: {PERIOD}s")

    seen = set()
    for c in (await q.get_candles(PAIR, time.time(), 20 * PERIOD, PERIOD)) or []:
        seen.add(c["time"])
        show(c)

    print("--- live, one line per closed candle (Ctrl+C to stop) ---")
    while True:
        await asyncio.sleep(max(5, PERIOD // 6))
        try:
            candles = await q.get_candles(PAIR, time.time(), 3 * PERIOD, PERIOD)
        except Exception as exc:  # noqa: BLE001 - keep the stream alive
            print("fetch error:", type(exc).__name__, exc)
            continue
        for c in candles or []:
            if c["time"] not in seen and c.get("close") is not None:
                seen.add(c["time"])
                show(c)


asyncio.run(main())
