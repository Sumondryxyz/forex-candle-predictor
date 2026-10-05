#!/usr/bin/env python3
"""Next-candle UP/DOWN signals from live Quotex candles, printed in the terminal.

Thin CLI over ``predictor.engine`` (numpy only, so it runs on Termux). It logs
in with ``QUOTEX_EMAIL``/``QUOTEX_PASSWORD``, loads history, prints a
walk-forward hit rate, then prints a signal per closed candle with a WIN/LOSS
check once the candle resolves. It never places a trade.

    export QUOTEX_EMAIL=... QUOTEX_PASSWORD=...
    python quotex_signals.py
"""
import asyncio
import os
import time
from collections import deque
from datetime import datetime, timezone

from pyquotex.stable_api import Quotex

from predictor import engine, quotex_session

HOST = os.getenv("QUOTEX_HOST", "market-qx.trade")
PAIR = os.getenv("PAIR", "EURUSD")
PERIOD = int(os.getenv("PERIOD", "60"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.60"))
HISTORY_HOURS = int(os.getenv("HISTORY_HOURS", "72"))
RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY", "60"))
PAYOUT = float(os.getenv("PAYOUT", "0.80"))

features = engine.features
train = engine.train
walk_forward = engine.walk_forward
arrays = engine.arrays


def stamp(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


async def main():
    quotex_session.apply(HOST)
    proxy = os.getenv("QUOTEX_PROXY", "").strip()
    q = Quotex(
        email=os.environ["QUOTEX_EMAIL"],
        password=os.environ["QUOTEX_PASSWORD"],
        lang="en",
        host=HOST,
        proxies={"http": proxy, "https": proxy} if proxy else None,
    )
    q.set_account_mode("PRACTICE")
    ok, reason = await q.connect()
    print("connected" if ok else f"failed: {reason}")
    if not ok:
        return

    print(f"balance: {await q.get_balance()}   {PAIR}   {PERIOD}s   min conf {MIN_CONF}")
    print(f"loading {HISTORY_HOURS}h of history ...")
    candles = await q.get_candles(PAIR, time.time(), HISTORY_HOURS * 3600, PERIOD)
    candles = [c for c in (candles or []) if c.get("close") is not None]
    candles.sort(key=lambda c: c["time"])
    print(f"history candles: {len(candles)}")
    if len(candles) < 500:
        print("not enough history to train; raise HISTORY_HOURS")
        return

    o, h, low, c, t = arrays(candles)
    X = features(o, h, low, c, t)

    report = walk_forward(X, c, min_conf=MIN_CONF, payout=PAYOUT)
    if report:
        print(
            f"walk-forward: {report['rate']:.1f}% hit on {report['signals']} signals "
            f"({report['coverage']:.0f}% of candles), ROI {report['roi']:+.1f}%/trade "
            f"at {PAYOUT:.0%} payout"
        )
        if report["rate"] < 100 / (1 + PAYOUT):
            print("  below break-even here — paper trade before risking money")
    else:
        print("not enough data for a walk-forward check")

    model = train(X, c)
    if model is None:
        print("training failed")
        return

    print("--- live signals, Ctrl+C to stop ---")
    seen = {x["time"] for x in candles}
    pending = None
    since_retrain = 0
    results = deque(maxlen=200)

    while True:
        await asyncio.sleep(max(5, PERIOD // 6))
        try:
            fresh = await q.get_candles(PAIR, time.time(), 10 * PERIOD, PERIOD)
        except Exception as exc:  # noqa: BLE001 - keep the stream alive
            print("fetch error:", type(exc).__name__, exc)
            continue

        for candle in sorted(fresh or [], key=lambda x: x["time"]):
            if candle["time"] in seen or candle.get("close") is None:
                continue
            seen.add(candle["time"])
            candles.append(candle)
            candles = candles[-6000:]
            since_retrain += 1

            if pending:
                went_up = candle["close"] > pending["ref_close"]
                win = went_up == (pending["dir"] == "UP")
                results.append(win)
                rate = sum(results) / len(results) * 100
                print(
                    f"    -> {'WIN ' if win else 'LOSS'} (close {candle['close']:.5f})"
                    f"   session {sum(results)}/{len(results)} = {rate:.0f}%"
                )
                pending = None

            o, h, low, c, t = arrays(candles)
            X = features(o, h, low, c, t)

            if since_retrain >= RETRAIN_EVERY:
                refreshed = train(X, c)
                if refreshed is not None:
                    model = refreshed
                    print(f"[retrained on {len(candles)} candles]")
                since_retrain = 0

            prob = float(model.prob(X[-1:])[0])
            direction = "UP" if prob > 0.5 else "DOWN"
            conf = max(prob, 1 - prob)
            entry = stamp(candle["time"] + PERIOD)

            if conf >= MIN_CONF:
                side = "CALL" if direction == "UP" else "PUT"
                print(f"{entry}  SIGNAL {direction:4s} ({side})  conf {conf * 100:.1f}%")
                pending = {"dir": direction, "ref_close": candle["close"]}
            else:
                print(f"{entry}  skip  ({direction} {conf * 100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
