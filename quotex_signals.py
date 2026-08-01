#!/usr/bin/env python3
"""Next-candle UP/DOWN signals from live Quotex candles, printed in the terminal.

Self-contained on purpose: only numpy is required, so it runs on Termux where
scipy/scikit-learn cannot be built.

How a signal is produced:

* 22 causal features (returns, wicks, EMA distances, RSI, MACD, ATR-normalised
  moves, Bollinger position, range position, streak, time of day);
* an ensemble of logistic regressions trained on different slices of history,
  averaged, so a single unlucky window cannot dominate;
* probabilities calibrated against a held-out tail of the history, so a printed
  62% means the model really hit ~62% on candles it had not seen;
* a walk-forward check that reports the honest hit rate before trading starts;
* the model retrains every RETRAIN_EVERY candles as new ones arrive.

    export QUOTEX_EMAIL=... QUOTEX_PASSWORD=...
    python quotex_signals.py
"""
import asyncio
import os
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
from pyquotex.stable_api import Quotex

from predictor import quotex_session

HOST = os.getenv("QUOTEX_HOST", "market-qx.trade")
PAIR = os.getenv("PAIR", "EURUSD")
PERIOD = int(os.getenv("PERIOD", "60"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.60"))
HISTORY_HOURS = int(os.getenv("HISTORY_HOURS", "72"))
RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY", "60"))
PAYOUT = float(os.getenv("PAYOUT", "0.80"))


# ------------------------------------------------------------------ indicators
def ema(values, span):
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rolling(values, window, fn):
    out = np.zeros_like(values)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = fn(values[lo : i + 1])
    return out


def rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = ema(np.clip(delta, 0, None), period)
    loss = ema(np.clip(-delta, 0, None), period)
    rs = gain / np.where(loss == 0, 1e-12, loss)
    return 100 - 100 / (1 + rs)


def features(o, h, low, c, times=None):
    """Causal features: row i only uses candles up to and including i."""
    eps = 1e-12
    rng = np.where(h - low == 0, eps, h - low)
    safe_c = np.where(c == 0, eps, c)

    ret1 = np.diff(c, prepend=c[0]) / safe_c
    ret3 = (c - np.roll(c, 3)) / safe_c
    ret5 = (c - np.roll(c, 5)) / safe_c
    ret10 = (c - np.roll(c, 10)) / safe_c

    ema_fast, ema_slow = ema(c, 5), ema(c, 20)
    ema_slowest = ema(c, 50)
    macd = ema(c, 12) - ema(c, 26)
    macd_hist = macd - ema(macd, 9)
    r = rsi(c)
    r_fast = rsi(c, 5)

    true_range = np.maximum(h - low, np.abs(np.diff(c, prepend=c[0])))
    atr = ema(true_range, 14)
    safe_atr = np.where(atr == 0, eps, atr)

    sma20 = rolling(c, 20, np.mean)
    std20 = rolling(c, 20, np.std)
    boll = (c - sma20) / np.where(std20 == 0, eps, std20)

    hi20 = rolling(h, 20, np.max)
    lo20 = rolling(low, 20, np.min)
    pos20 = (c - lo20) / np.where(hi20 - lo20 == 0, eps, hi20 - lo20)

    body = (c - o) / rng
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - low) / rng
    body_atr = (c - o) / safe_atr
    range_atr = rng / safe_atr

    up = (c > o).astype(float)
    streak = np.zeros_like(c)
    for i in range(1, len(c)):
        streak[i] = streak[i - 1] + 1 if up[i] == up[i - 1] else 0
    streak *= np.where(up == 1, 1, -1)

    vol_ratio = rolling(np.abs(ret1), 5, np.mean) / (
        rolling(np.abs(ret1), 30, np.mean) + eps
    )

    if times is None:
        sin_t = np.zeros_like(c)
        cos_t = np.zeros_like(c)
    else:
        frac = (np.asarray(times, float) % 86400) / 86400 * 2 * np.pi
        sin_t, cos_t = np.sin(frac), np.cos(frac)

    cols = [
        ret1 * 1e4,
        ret3 * 1e4,
        ret5 * 1e4,
        ret10 * 1e4,
        body,
        upper,
        lower,
        body_atr,
        range_atr,
        (c - ema_fast) / safe_atr,
        (c - ema_slow) / safe_atr,
        (c - ema_slowest) / safe_atr,
        (ema_fast - ema_slow) / safe_atr,
        (r - 50) / 50,
        (r_fast - 50) / 50,
        macd_hist / safe_atr,
        boll,
        pos20 - 0.5,
        streak / 5,
        vol_ratio - 1,
        sin_t,
        cos_t,
    ]
    X = np.column_stack(cols)
    X[:50] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------- model
def fit_one(X, y, epochs=600, lr=0.15, l2=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / std
    w = rng.normal(0, 0.01, Z.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        err = p - y
        w -= lr * (Z.T @ err / len(y) + l2 * w)
        b -= lr * err.mean()
    return w, b, mean, std


def raw_prob(member, X):
    w, b, mean, std = member
    z = (X - mean) / std
    return 1 / (1 + np.exp(-np.clip(z @ w + b, -30, 30)))


class Ensemble:
    """Bagged logistic regressions plus a bin-based probability calibration."""

    def __init__(self, members, calib=None):
        self.members = members
        self.calib = calib or {}

    def raw(self, X):
        X = np.atleast_2d(X)
        return np.mean([raw_prob(m, X) for m in self.members], axis=0)

    def prob(self, X):
        p = self.raw(X)
        if not self.calib:
            return p
        edges, values = self.calib["edges"], self.calib["values"]
        idx = np.clip(np.digitize(p, edges) - 1, 0, len(values) - 1)
        return values[idx]


def labels(c):
    y = (np.roll(c, -1) > c).astype(float)
    usable = np.roll(c, -1) != c
    usable[-1] = False
    return y, usable


def train(X, c, seeds=5):
    """Train on the first 80%, calibrate on the last 20% (never seen in fit)."""
    y, usable = labels(c)
    idx = np.flatnonzero(usable)
    if len(idx) < 300:
        return None

    split = int(len(idx) * 0.8)
    fit_idx, cal_idx = idx[:split], idx[split:]

    members = []
    n = len(fit_idx)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        sample = rng.choice(fit_idx, size=n, replace=True) if s else fit_idx
        members.append(fit_one(X[sample], y[sample], seed=s))
    model = Ensemble(members)

    p_cal = model.raw(X[cal_idx])
    y_cal = y[cal_idx]
    edges = np.linspace(0.3, 0.7, 9)
    values = []
    for i in range(len(edges)):
        lo = edges[i]
        hi = edges[i + 1] if i + 1 < len(edges) else 1.1
        in_bin = (p_cal >= lo) & (p_cal < hi)
        values.append(y_cal[in_bin].mean() if in_bin.sum() >= 15 else np.nan)
    values = np.array(values, float)
    # bins without enough samples fall back to the raw bin centre
    centres = np.append(edges[1:], 1.0)
    values = np.where(np.isnan(values), np.clip(centres - 0.02, 0.05, 0.95), values)
    model.calib = {"edges": edges, "values": values}
    return model


def walk_forward(X, c, folds=4):
    """Honest out-of-sample check: train on the past, score the next slice."""
    y, usable = labels(c)
    idx = np.flatnonzero(usable)
    if len(idx) < 800:
        return None

    hits, signals, total = 0, 0, 0
    step = len(idx) // (folds + 1)
    for f in range(1, folds + 1):
        train_idx = idx[: step * f]
        test_idx = idx[step * f : step * (f + 1)]
        model = train(X[train_idx], c[train_idx])
        if model is None:
            continue
        p = model.prob(X[test_idx])
        conf = np.maximum(p, 1 - p)
        picked = conf >= MIN_CONF
        total += len(test_idx)
        signals += int(picked.sum())
        hits += int(((p[picked] > 0.5).astype(float) == y[test_idx][picked]).sum())

    if signals == 0:
        return {"signals": 0, "rate": 0.0, "coverage": 0.0, "roi": 0.0}
    rate = hits / signals
    return {
        "signals": signals,
        "rate": rate * 100,
        "coverage": signals / max(total, 1) * 100,
        "roi": (rate * PAYOUT - (1 - rate)) * 100,
    }


# ------------------------------------------------------------------------ main
def stamp(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


def arrays(candles):
    o = np.array([x["open"] for x in candles], float)
    h = np.array([x["high"] for x in candles], float)
    low = np.array([x["low"] for x in candles], float)
    c = np.array([x["close"] for x in candles], float)
    t = np.array([x["time"] for x in candles], float)
    return o, h, low, c, t


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

    report = walk_forward(X, c)
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
