#!/usr/bin/env python3
"""Rule-based edge test on clean Dukascopy 1m data.

No training, no leakage: each rule reads only candles up to and including the
current one and predicts the next candle's direction. Scored on real moves only
(flat candles excluded), weekdays, overall and by session. A rule is an edge
only when its Wilson 95% lower bound clears the 55.56% break-even.

    python research_rules.py --pairs EURUSD,GBPUSD
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from predictor import readiness

BREAK_EVEN = 1 / (1 + 0.80)
SESSIONS = {"asia": (0, 7), "london": (7, 12), "ny": (12, 16), "late": (16, 21)}


def _stats(pred, y, n):
    """pred: +1/-1/0 (0 = no signal); y: +1/-1 next move."""
    m = pred != 0
    if m.sum() < 30:
        return None
    wins = int((pred[m] == y[m]).sum())
    n = int(m.sum())
    lo = readiness.wilson_lower_bound(wins, n) or 0.0
    p = readiness.p_value(wins, n, BREAK_EVEN)
    return wins / n, n, lo, p


def test(df, name, horizon=1):

    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    hours = np.array([x.hour for x in df.index])
    dow = np.array([x.dayofweek for x in df.index])
    nxt = np.roll(c, -horizon)
    y = np.sign(nxt - c).astype(int)
    valid = (nxt != c) & (dow < 5)
    valid[-horizon:] = False

    ret1 = np.zeros_like(c)
    ret1[1:] = np.diff(c)
    d = np.sign(ret1).astype(int)

    # consecutive streak of same-direction candles
    streak = np.zeros_like(d)
    for i in range(1, len(d)):
        streak[i] = streak[i - 1] + d[i] if d[i] == d[i - 1] and d[i] != 0 else d[i]

    tr = np.maximum(h - l, np.abs(h - np.r_[c[0], c[:-1]]))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()

    rules = {}
    # 1. trend-follow last 3
    rules["trend3"] = np.where(streak >= 3, 1, np.where(streak <= -3, -1, 0))
    # 2. fade last 3 (mean reversion)
    rules["fade3"] = np.where(streak >= 3, -1, np.where(streak <= -3, 1, 0))
    # 3. momentum over last 5 with threshold (vol filter)
    mom5 = np.zeros_like(c)
    mom5[5:] = c[5:] - c[:-5]
    thr = 3 * atr
    rules["mom5"] = np.where(mom5 > thr, 1, np.where(mom5 < -thr, -1, 0))
    # 4. volatility breakout: big candle -> follow it
    big = (h - l) > 2 * atr
    rules["breakout"] = np.where(big, d, 0)

    print(f"\n=== {name} (n={int(valid.sum())} valid moves) ===")
    for rname, pred in rules.items():
        pred = np.where(valid, pred, 0)
        res = _stats(pred, y, valid.sum())
        if res is None:
            print(f"  {rname:9s}: too few signals")
            continue
        hit, n, lo, p = res
        tag = "EDGE" if lo > BREAK_EVEN else ""
        print(f"  {rname:9s}: {100*hit:5.2f}%  n={n:6d}  wilson_lo={100*lo:5.2f}%  "
              f"p={p:.3f}  {tag}")
        # session breakdown for this rule
        for sname, (a, b) in SESSIONS.items():
            sm = (hours >= a) & (hours < b)
            pr = np.where(sm, pred, 0)
            r = _stats(pr, y, int(valid.sum()))
            if r is None:
                continue
            hit2, n2, lo2, p2 = r
            if lo2 > BREAK_EVEN:
                print(f"      {sname:6s}: {100*hit2:5.2f}%  n={n2:5d}  "
                      f"wilson_lo={100*lo2:5.2f}%  p={p2:.3f}  EDGE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="var/duka_1m.pkl")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD")
    args = ap.parse_args()
    with open(args.pickle, "rb") as fh:
        data = pickle.load(fh)
    for pair in [p for p in args.pairs.split(",") if p in data]:
        test(data[pair], pair)


if __name__ == "__main__":
    main()
