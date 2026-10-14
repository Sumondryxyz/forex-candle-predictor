#!/usr/bin/env python3
"""Edge hunt v3: honest scoring on real moves only.

v2 scored every candle with ``next_close > close``. Weekend candles are flat
(close == next close) ~35% of the time, and a flat candle is counted as a
"down" win even though a real broker refunds/ties it. That inflates the hit
rate. This version scores only candles where the next close actually moved, and
optionally drops weekend hours, then reports per-fold consistency so a single
lucky fold cannot masquerade as an edge.

    python research_edge_v3.py --intervals 1m --pairs EURUSD,GBPUSD
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from predictor import readiness
from research_edge_v2 import _resample, rich_features

BREAK_EVEN = 1 / (1 + 0.80)


def walk_forward(X, c, ts, n_folds=6, train_frac=0.6, active_only=True,
                 drop_weekend=True):
    import lightgbm as lgb

    nxt = np.roll(c, -1)
    moved = nxt != c
    y = (nxt > c).astype(int)

    ok = np.ones(len(c), bool)
    ok[-1] = False
    if active_only:
        ok &= moved
    if drop_weekend:
        dow = np.array([pd.Timestamp(t).dayofweek for t in ts])
        ok &= dow < 5

    idx = np.flatnonzero(ok)
    start = int(len(idx) * train_frac)
    if len(idx) - start < 200:
        return None
    fold = max(1, (len(idx) - start) // n_folds)

    rows = []
    for s in range(start, len(idx), fold):
        e = min(s + fold, len(idx))
        tr, te = idx[:s], idx[s:e]
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                               num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                               min_child_samples=50, random_state=0, verbose=-1)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        pred_up = p > 0.5
        correct = (pred_up == (y[te] == 1))
        rows.append(pd.DataFrame({"conf": np.maximum(p, 1 - p),
                                  "correct": correct.astype(float),
                                  "fold": len(rows)}))
    return pd.concat(rows, ignore_index=True)


def report(pair, res):
    if res is None:
        print(f"  {pair}: not enough data")
        return
    print(f"  {pair}: n={len(res)}  all={100*res['correct'].mean():.2f}%")
    for floor in (0.55, 0.60, 0.65, 0.70):
        sel = res[res["conf"] >= floor]
        if len(sel) < 20:
            continue
        wins, n = sel["correct"].sum(), len(sel)
        lo = readiness.wilson_lower_bound(int(wins), int(n)) or 0.0
        p = readiness.p_value(int(wins), int(n), BREAK_EVEN)
        # per-fold hit rates
        folds = [f"{100*g['correct'].mean():.0f}%({len(g)})"
                 for _, g in sel.groupby("fold")]
        tag = "EDGE" if lo > BREAK_EVEN else "not yet"
        print(f"      conf>={floor}: {100*sel['correct'].mean():5.2f}%  n={n:5d}  "
              f"wilson_lo={100*lo:5.2f}%  p={p:.3f}  {tag}  folds={folds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="var/duka_1m.pkl")
    ap.add_argument("--intervals", default="1m")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD")
    args = ap.parse_args()

    with open(args.pickle, "rb") as fh:
        data = pickle.load(fh)
    pairs = [p for p in args.pairs.split(",") if p in data]

    for interval in args.intervals.split(","):
        print(f"\n===== {interval} (real moves only, weekdays) =====")
        for pair in pairs:
            df = _resample(data[pair], interval)
            X = rich_features(df)
            c = df["close"].to_numpy(float)
            ts = np.array([x.timestamp() for x in df.index], float)
            report(pair, walk_forward(X, c, ts))


if __name__ == "__main__":
    main()
