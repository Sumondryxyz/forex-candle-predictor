#!/usr/bin/env python3
"""Next-candle UP/DOWN signals from live Quotex candles, printed in the terminal.

Self-contained on purpose: only numpy is required, so it runs on Termux where
scipy/scikit-learn cannot be built. The model is a logistic regression trained
by gradient descent on the pair's own recent history, then applied to each
freshly closed candle. Signals below MIN_CONF are shown as SKIP.

    export QUOTEX_EMAIL=... QUOTEX_PASSWORD=...
    python quotex_signals.py
"""
import asyncio
import os
import time
from datetime import datetime, timezone

import numpy as np
from pyquotex.stable_api import Quotex

from predictor import quotex_session

HOST = os.getenv("QUOTEX_HOST", "market-qx.trade")
PAIR = os.getenv("PAIR", "EURUSD")
PERIOD = int(os.getenv("PERIOD", "60"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.60"))
HISTORY_HOURS = int(os.getenv("HISTORY_HOURS", "24"))




# ------------------------------------------------------------------- features
def ema(values, span):
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = ema(np.clip(delta, 0, None), period)
    loss = ema(np.clip(-delta, 0, None), period)
    rs = gain / np.where(loss == 0, 1e-12, loss)
    return 100 - 100 / (1 + rs)


def features(o, h, low, c):
    """Causal features: row i uses candles up to i only."""
    eps = 1e-12
    rng = np.where(h - low == 0, eps, h - low)
    ret1 = np.diff(c, prepend=c[0]) / np.where(c == 0, eps, c)
    ret3 = (c - np.roll(c, 3)) / np.where(c == 0, eps, c)
    ret5 = (c - np.roll(c, 5)) / np.where(c == 0, eps, c)

    ema_fast, ema_slow = ema(c, 5), ema(c, 20)
    macd = ema(c, 12) - ema(c, 26)
    macd_hist = macd - ema(macd, 9)
    r = rsi(c)

    body = (c - o) / rng
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - low) / rng

    up = (c > o).astype(float)
    streak = np.zeros_like(c)
    for i in range(1, len(c)):
        streak[i] = streak[i - 1] + 1 if up[i] == up[i - 1] else 0
    streak *= np.where(up == 1, 1, -1)

    cols = [
        ret1 * 1e4,
        ret3 * 1e4,
        ret5 * 1e4,
        body,
        upper,
        lower,
        (c - ema_fast) / np.where(ema_fast == 0, eps, ema_fast) * 1e4,
        (c - ema_slow) / np.where(ema_slow == 0, eps, ema_slow) * 1e4,
        (ema_fast - ema_slow) / np.where(ema_slow == 0, eps, ema_slow) * 1e4,
        (r - 50) / 50,
        macd_hist * 1e4,
        streak / 5,
    ]
    X = np.column_stack(cols)
    X[:6] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def fit(X, y, epochs=400, lr=0.1, l2=1e-3):
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / std
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        err = p - y
        w -= lr * (Z.T @ err / len(y) + l2 * w)
        b -= lr * err.mean()
    return w, b, mean, std


def predict(model, x):
    w, b, mean, std = model
    z = (x - mean) / std
    return float(1 / (1 + np.exp(-(z @ w + b))))


def train_from(candles):
    o = np.array([c["open"] for c in candles], float)
    h = np.array([c["high"] for c in candles], float)
    low = np.array([c["low"] for c in candles], float)
    c = np.array([c["close"] for c in candles], float)
    X = features(o, h, low, c)
    y = (np.roll(c, -1) > c).astype(float)
    keep = np.roll(c, -1) != c
    keep[-1] = False
    return fit(X[keep], y[keep]), X, c


def backtest(model, X, c):
    """Honest in-sample check on the same history the model was trained on."""
    y = (np.roll(c, -1) > c).astype(float)[:-1]
    probs = np.array([predict(model, x) for x in X[:-1]])
    conf = np.maximum(probs, 1 - probs)
    picked = conf >= MIN_CONF
    if picked.sum() == 0:
        return 0, 0.0
    correct = ((probs[picked] > 0.5).astype(float) == y[picked]).mean()
    return int(picked.sum()), float(correct * 100)


# ----------------------------------------------------------------------- main
def stamp(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


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

    print(f"balance: {await q.get_balance()}   {PAIR}   {PERIOD}s   min conf {MIN_CONF}")
    print("loading history ...")
    candles = await q.get_candles(PAIR, time.time(), HISTORY_HOURS * 3600, PERIOD)
    candles = [c for c in (candles or []) if c.get("close") is not None]
    print(f"history candles: {len(candles)}")
    if len(candles) < 200:
        print("not enough history to train, try a bigger HISTORY_HOURS")
        return

    model, X, closes = train_from(candles)
    signals, hit = backtest(model, X, closes)
    print(f"in-sample check: {signals} signals at >= {MIN_CONF} confidence, {hit:.1f}% hit")
    print("(in-sample numbers flatter the model; treat live results as the truth)")
    print("--- live signals, Ctrl+C to stop ---")

    seen = {c["time"] for c in candles}
    pending = None

    while True:
        await asyncio.sleep(max(5, PERIOD // 6))
        try:
            fresh = await q.get_candles(PAIR, time.time(), 10 * PERIOD, PERIOD)
        except Exception as exc:  # noqa: BLE001 - keep running
            print("fetch error:", type(exc).__name__, exc)
            continue

        for c in fresh or []:
            if c["time"] in seen or c.get("close") is None:
                continue
            seen.add(c["time"])
            candles.append(c)
            candles = candles[-5000:]

            if pending:
                went_up = c["close"] > pending["ref_close"]
                result = "WIN " if went_up == (pending["dir"] == "UP") else "LOSS"
                print(f"    -> {result} (close {c['close']:.5f})")
            pending = None

            o = np.array([x["open"] for x in candles], float)
            h = np.array([x["high"] for x in candles], float)
            low = np.array([x["low"] for x in candles], float)
            cl = np.array([x["close"] for x in candles], float)
            prob = predict(model, features(o, h, low, cl)[-1])
            direction = "UP" if prob > 0.5 else "DOWN"
            conf = max(prob, 1 - prob)

            if conf >= MIN_CONF:
                print(
                    f"{stamp(c['time'] + PERIOD)}  SIGNAL {direction:4s} "
                    f"({'CALL' if direction == 'UP' else 'PUT'})  conf {conf * 100:.1f}%"
                )
                pending = {"dir": direction, "ref_close": c["close"]}
            else:
                print(f"{stamp(c['time'] + PERIOD)}  skip  ({direction} {conf * 100:.1f}%)")


asyncio.run(main())
