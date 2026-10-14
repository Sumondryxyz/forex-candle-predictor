#!/usr/bin/env python3
"""Sharpen the mean-reversion signal.

Every simple rule was ~50% except fading a run of same-direction candles
("fade"). This sweeps the fade: run length, how far price must have stretched
from its short mean (in ATR units), session, and expiry horizon -- pooled over
pairs for sample size. Honest: only past candles are read, flat candles and
weekends are excluded, and an edge needs its Wilson lower bound above 55.56%.

    python research_fade.py --pairs EURUSD,GBPUSD
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from predictor import readiness
from research_edge_v2 import _resample

BREAK_EVEN = 1 / (1 + 0.80)
SESSIONS = {"asia": (0, 7), "london": (7, 12), "ny": (12, 16), "late": (16, 21)}


def _score(pred, y):
    m = pred != 0
    if m.sum() < 100:
        return None
    wins, n = int((pred[m] == y[m]).sum()), int(m.sum())
    lo = readiness.wilson_lower_bound(wins, n) or 0.0
    p = readiness.p_value(wins, n, BREAK_EVEN)
    return wins / n, n, lo, p


def sweep(df, name, horizons=(1, 2, 3, 5)):
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    hours = np.array([x.hour for x in df.index])
    dow = np.array([x.dayofweek for x in df.index])

    ret1 = np.zeros_like(c)
    ret1[1:] = np.diff(c)
    d = np.sign(ret1).astype(int)
    streak = np.zeros_like(d)
    for i in range(1, len(d)):
        streak[i] = streak[i - 1] + d[i] if d[i] == d[i - 1] and d[i] != 0 else d[i]

    tr = np.maximum(h - l, np.abs(h - np.r_[c[0], c[:-1]]))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    ma = pd.Series(c).rolling(20).mean().to_numpy()
    stretch = (c - ma) / (atr + 1e-12)

    print(f"\n=== {name} ===")
    for horizon in horizons:
        nxt = np.roll(c, -horizon)
        y = np.sign(nxt - c).astype(int)
        valid = (nxt != c) & (dow < 5)
        valid[-horizon:] = False

        for run in (2, 3, 4, 5):
            base = np.where(streak >= run, -1, np.where(streak <= -run, 1, 0))
            for k in (0.0, 0.5, 1.0):
                over = np.abs(stretch) >= k
                pred = np.where(over & valid, base, 0)
                res = _score(pred, y)
                if res is None:
                    continue
                hit, n, lo, p = res
                if lo <= BREAK_EVEN and hit < 0.53:
                    continue  # only surface anything interesting
                tag = "EDGE" if lo > BREAK_EVEN else ""
                print(f"  h={horizon} run>={run} stretch>={k}: {100*hit:5.2f}%  "
                      f"n={n:5d}  wilson_lo={100*lo:5.2f}%  p={p:.3f}  {tag}")
                for sname, (a, b) in SESSIONS.items():
                    sm = (hours >= a) & (hours < b)
                    r = _score(np.where(sm, pred, 0), y)
                    if r and r[2] > BREAK_EVEN:
                        print(f"      {sname:6s}: {100*r[0]:5.2f}%  n={r[1]:5d}  "
                              f"wilson_lo={100*r[2]:5.2f}%  p={r[3]:.3f}  EDGE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="var/duka_1m.pkl")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD")
    ap.add_argument("--interval", default="1m")
    args = ap.parse_args()
    with open(args.pickle, "rb") as fh:
        data = pickle.load(fh)
    for pair in [p for p in args.pairs.split(",") if p in data]:
        sweep(_resample(data[pair], args.interval), pair)


if __name__ == "__main__":
    main()
