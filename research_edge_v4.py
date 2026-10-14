#!/usr/bin/env python3
"""Edge hunt v4: add higher-timeframe context and pool pairs.

v2/v3 used only features computed on the same candle stream. This adds the
higher-timeframe view (15m/1h trend, RSI, ATR) forward-filled onto the base
candles -- the "multi-timeframe" information a human trader uses -- and can pool
several pairs into one training set to raise the sample size. Scoring is the
same honest walk-forward: real moves only, weekdays, Wilson lower bound vs the
55.56% break-even.

    python research_edge_v4.py --interval 1m --pairs EURUSD,GBPUSD
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from predictor import readiness
from research_edge_v2 import _resample, rich_features

BREAK_EVEN = 1 / (1 + 0.80)
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum"}


def htf_context(df: pd.DataFrame, rules=("15min", "1h")) -> np.ndarray:
    base = rich_features(df)
    blocks = [base]
    for rule in rules:
        # label/closed="right" stamps each bar with its CLOSE time, so a bar is
        # only visible from the minute it finishes. Without this, ffill would
        # hand a 12:07 candle the 12:00-12:14 bar whose close is 12:14 -- future
        # information that inflates accuracy (a classic look-ahead leak).
        htf = df.resample(rule, label="right", closed="right").agg(_AGG).dropna()
        if len(htf) < 50:
            continue
        hf = rich_features(htf)
        aligned = pd.DataFrame(hf, index=htf.index).reindex(df.index, method="ffill")
        blocks.append(aligned.to_numpy())
    return np.nan_to_num(np.column_stack(blocks), nan=0.0, posinf=0.0, neginf=0.0)


def walk_forward(X, c, ts, n_folds=8, train_frac=0.6):
    import lightgbm as lgb

    nxt = np.roll(c, -1)
    y = (nxt > c).astype(int)
    dow = np.array([pd.Timestamp(t).dayofweek for t in ts])
    ok = (nxt != c) & (dow < 5)
    ok[-1] = False
    idx = np.flatnonzero(ok)
    start = int(len(idx) * train_frac)
    if len(idx) - start < 200:
        return None
    fold = max(1, (len(idx) - start) // n_folds)
    rows = []
    for s in range(start, len(idx), fold):
        e = min(s + fold, len(idx))
        tr, te = idx[:s], idx[s:e]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                               num_leaves=15, subsample=0.8, colsample_bytree=0.7,
                               min_child_samples=100, reg_lambda=1.0,
                               random_state=0, verbose=-1)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        rows.append(pd.DataFrame({"conf": np.maximum(p, 1 - p),
                                  "correct": (p > 0.5) == (y[te] == 1),
                                  "fold": len(rows)}))
    return pd.concat(rows, ignore_index=True)


def report(name, res):
    if res is None:
        print(f"  {name}: not enough data")
        return
    print(f"  {name}: n={len(res)}  all={100*res['correct'].mean():.2f}%")
    for floor in (0.55, 0.60, 0.65, 0.70):
        sel = res[res["conf"] >= floor]
        if len(sel) < 30:
            continue
        wins, n = int(sel["correct"].sum()), len(sel)
        lo = readiness.wilson_lower_bound(wins, n) or 0.0
        p = readiness.p_value(wins, n, BREAK_EVEN)
        folds = [f"{100*g['correct'].mean():.0f}%" for _, g in sel.groupby("fold")]
        tag = "EDGE" if lo > BREAK_EVEN else "not yet"
        print(f"      conf>={floor}: {100*sel['correct'].mean():5.2f}%  n={n:5d}  "
              f"wilson_lo={100*lo:5.2f}%  p={p:.3f}  {tag}  folds={folds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="var/duka_1m.pkl")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD")
    args = ap.parse_args()

    with open(args.pickle, "rb") as fh:
        data = pickle.load(fh)
    pairs = [p for p in args.pairs.split(",") if p in data]

    print(f"\n===== {args.interval} + 15m/1h context (real moves, weekdays) =====")
    for pair in pairs:
        df = _resample(data[pair], args.interval)
        X = htf_context(df)
        c = df["close"].to_numpy(float)
        ts = np.array([x.timestamp() for x in df.index], float)
        report(pair, walk_forward(X, c, ts))


if __name__ == "__main__":
    main()
