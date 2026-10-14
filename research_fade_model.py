#!/usr/bin/env python3
"""Focus a model on the mean-reversion setup, pooled across pairs.

The fade sweep showed the only repeatable signal: after a run of >=3-5
same-direction candles, price tends to revert (51-55% on the next move). This
trains a GBM only on those setup candles, pools the pairs for sample size, and
checks consistency by month so a single regime cannot look like an edge.

    python research_fade_model.py --pairs EURUSD,GBPUSD --horizon 5
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from predictor import readiness
from research_edge_v2 import _resample, rich_features

BREAK_EVEN = 1 / (1 + 0.80)


def build(data, pairs, interval, run_min, horizon):
    Xs, ys, ts_all = [], [], []
    for pair in pairs:
        df = _resample(data[pair], interval)
        c = df["close"].to_numpy(float)
        ret1 = np.zeros_like(c)
        ret1[1:] = np.diff(c)
        d = np.sign(ret1).astype(int)
        streak = np.zeros_like(d)
        for i in range(1, len(d)):
            streak[i] = streak[i - 1] + d[i] if d[i] == d[i - 1] and d[i] != 0 else d[i]
        nxt = np.roll(c, -horizon)
        y = (nxt > c).astype(int)
        dow = np.array([x.dayofweek for x in df.index])
        ok = (nxt != c) & (dow < 5) & (np.abs(streak) >= run_min)
        ok[-horizon:] = False
        idx = np.flatnonzero(ok)
        Xs.append(rich_features(df)[idx])
        ys.append(y[idx])
        ts_all.append(np.array([df.index[i].timestamp() for i in idx], float))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    ts = np.concatenate(ts_all)
    order = np.argsort(ts)
    return X[order], y[order], ts[order]


def walk_forward(X, y, ts, n_folds=8, train_frac=0.5):
    import lightgbm as lgb

    start = int(len(y) * train_frac)
    fold = max(1, (len(y) - start) // n_folds)
    rows = []
    for s in range(start, len(y), fold):
        e = min(s + fold, len(y))
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04,
                               num_leaves=15, subsample=0.8, colsample_bytree=0.7,
                               min_child_samples=100, reg_lambda=1.0,
                               random_state=0, verbose=-1)
        m.fit(X[:s], y[:s])
        p = m.predict_proba(X[s:e])[:, 1]
        rows.append(pd.DataFrame({"conf": np.maximum(p, 1 - p),
                                  "correct": (p > 0.5) == (y[s:e] == 1),
                                  "month": pd.to_datetime(ts[s:e], unit="s").strftime("%Y-%m")}))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="var/duka_1m.pkl")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--run", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    with open(args.pickle, "rb") as fh:
        data = pickle.load(fh)
    pairs = [p for p in args.pairs.split(",") if p in data]
    X, y, ts = build(data, pairs, args.interval, args.run, args.horizon)
    print(f"setup run>={args.run} horizon={args.horizon}  pooled n={len(y)}  "
          f"base-rate up={100*y.mean():.1f}%")

    res = walk_forward(X, y, ts)
    print(f"  all: {100*res['correct'].mean():.2f}%")
    for floor in (0.5, 0.55, 0.60, 0.65):
        sel = res[res["conf"] >= floor]
        if len(sel) < 30:
            continue
        wins, n = int(sel["correct"].sum()), len(sel)
        lo = readiness.wilson_lower_bound(wins, n) or 0.0
        p = readiness.p_value(wins, n, BREAK_EVEN)
        tag = "EDGE" if lo > BREAK_EVEN else ""
        print(f"  conf>={floor}: {100*sel['correct'].mean():5.2f}%  n={n:5d}  "
              f"wilson_lo={100*lo:5.2f}%  p={p:.3f}  {tag}")
        if floor >= 0.6:
            by_m = {m: f"{100*g['correct'].mean():.0f}%({len(g)})"
                    for m, g in sel.groupby("month")}
            print(f"      by month: {by_m}")


if __name__ == "__main__":
    main()
