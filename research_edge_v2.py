#!/usr/bin/env python3
"""Edge hunt v2: richer features + gradient boosting, honest walk-forward.

The v1 hunt used the live engine's 35 features and a linear model, and found no
1m/5m edge. This tests whether *more* information helps: a wide technical feature
set fed to LightGBM, evaluated with the same rolling walk-forward the live feed
uses, so a "win" here is a win the feed could actually take.

    python research_edge_v2.py --pickle var/duka_1m.pkl --intervals 1m,5m
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from predictor import readiness

_RESAMPLE = {"1m": None, "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}
BREAK_EVEN = 1 / (1 + 0.80)


def _resample(df, interval):
    rule = _RESAMPLE[interval]
    if rule is None:
        return df
    return (df.resample(rule)
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"})
              .dropna())


def rich_features(df: pd.DataFrame) -> np.ndarray:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(float)
    idx = df.index
    out = {}

    ret = np.zeros_like(c)
    ret[1:] = np.diff(c) / c[:-1]
    out["ret1"] = ret

    for w in (2, 3, 5, 10, 20, 60):
        out[f"mom{w}"] = pd.Series(c).pct_change(w).to_numpy()

    for w in (5, 10, 20, 60):
        out[f"vol{w}"] = pd.Series(ret).rolling(w).std().to_numpy()

    rng = h - l
    body = c - o
    out["body"] = body / (rng + 1e-12)
    out["upper_wick"] = (h - np.maximum(o, c)) / (rng + 1e-12)
    out["lower_wick"] = (np.minimum(o, c) - l) / (rng + 1e-12)
    out["range_rel"] = rng / (c + 1e-12)

    # RSI(14)
    delta = pd.Series(c).diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi14"] = (100 - 100 / (1 + up / (dn + 1e-12))).to_numpy()

    # ATR percentile (regime)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(np.maximum(h - l, np.abs(h - prev_c)), np.abs(l - prev_c))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_pct"] = atr.rolling(200).rank(pct=True).to_numpy()

    # Bollinger z
    ma20 = pd.Series(c).rolling(20).mean()
    sd20 = pd.Series(c).rolling(20).std()
    out["bb_z"] = ((pd.Series(c) - ma20) / (sd20 + 1e-12)).to_numpy()

    # MACD histogram
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()).to_numpy()

    # streak
    d = np.sign(ret)
    st = np.zeros_like(c)
    for i in range(1, len(c)):
        st[i] = st[i - 1] + d[i] if d[i] == d[i - 1] and d[i] != 0 else d[i]
    out["streak"] = st

    # volume z
    out["vol_z"] = ((v - pd.Series(v).rolling(60).mean()) /
                    (pd.Series(v).rolling(60).std() + 1e-12)).to_numpy()

    # clock
    secs = np.array([x.hour * 3600 + x.minute * 60 for x in idx], float)
    out["tod_sin"] = np.sin(2 * np.pi * secs / 86400)
    out["tod_cos"] = np.cos(2 * np.pi * secs / 86400)
    out["dow"] = np.array([x.dayofweek for x in idx], float)

    keys = list(out)
    X = np.column_stack([out[k] for k in keys])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def walk_forward(X, c, n_folds=6, train_frac=0.6):
    """Rolling-origin walk-forward. Returns per-signal conf + correctness."""
    import lightgbm as lgb

    y = (np.roll(c, -1) > c).astype(int)
    usable = np.roll(c, -1) != c
    usable[-1] = False
    idx = np.flatnonzero(usable)
    start = int(len(idx) * train_frac)
    if len(idx) - start < 200:
        return None
    fold = max(1, (len(idx) - start) // n_folds)

    probs, correct, confs = [], [], []
    for s in range(start, len(idx), fold):
        e = min(s + fold, len(idx))
        tr, te = idx[:s], idx[s:e]
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                               num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                               min_child_samples=50, random_state=0, verbose=-1)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        probs.append(p)
        correct.append((p > 0.5) == (y[te] == 1))
        confs.append(np.maximum(p, 1 - p))
    return (np.concatenate(probs), np.concatenate(correct).astype(float),
            np.concatenate(confs))


def _verdict(wins, n):
    if n == 0:
        return "no signals", 0.0, 1.0
    lo = readiness.wilson_lower_bound(int(wins), int(n)) or 0.0
    p = readiness.p_value(int(wins), int(n), BREAK_EVEN)
    return ("EDGE" if lo > BREAK_EVEN else "not yet"), lo, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default="var/duka_1m.pkl")
    ap.add_argument("--intervals", default="1m,5m")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY")
    args = ap.parse_args()

    with open(args.pickle, "rb") as fh:
        data = pickle.load(fh)
    pairs = [p for p in args.pairs.split(",") if p in data]

    for interval in args.intervals.split(","):
        print(f"\n===== {interval} =====")
        for pair in pairs:
            df = _resample(data[pair], interval)
            X = rich_features(df)
            c = df["close"].to_numpy(float)
            res = walk_forward(X, c)
            if res is None:
                print(f"  {pair}: not enough data")
                continue
            _, correct, conf = res
            print(f"  {pair}: n={len(correct)}  all={100*correct.mean():.2f}%  "
                  f"up-rate={100*(c[1:]>c[:-1]).mean():.1f}%")
            for floor in (0.55, 0.60, 0.65, 0.70):
                sel = conf >= floor
                if sel.sum() < 30:
                    continue
                wins = correct[sel].sum()
                tag, lo, p = _verdict(wins, sel.sum())
                print(f"      conf>={floor}: {100*correct[sel].mean():5.2f}%  "
                      f"n={int(sel.sum()):5d}  wilson_lo={100*lo:5.2f}%  "
                      f"p={p:.3f}  {tag}")


if __name__ == "__main__":
    main()
