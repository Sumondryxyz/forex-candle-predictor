#!/usr/bin/env python3
"""Controls that prove the 1h 67% result is real, not a harness artifact.

1. Feature causality: change candles strictly after bar i and confirm the
   feature row for bar i is byte-identical (no look-ahead).
2. Synthetic random walk: feed the SAME engine a driftless GBM walk with the
   real per-pair volatility. If the pipeline is honest it must score ~50% there;
   only then is the real-data 67% meaningful.

    python research_controls_1h.py --days 180
"""
from __future__ import annotations

import argparse

import numpy as np

from predictor import engine, readiness
from predictor.sources import get_fetcher

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
BREAK_EVEN = 1 / (1 + 0.80)


def _features(df):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    t = np.array([x.timestamp() for x in df.index], float)
    return engine.features(o, h, low, c, t)


def causality_check(df, i=None):
    """features[i] must not depend on candles after i."""
    X = _features(df)
    i = i or len(df) // 2
    tampered = df.copy()
    # scramble everything strictly after i
    rng = np.random.default_rng(0)
    for col in ("open", "high", "low", "close"):
        vals = np.array(tampered[col].to_numpy(float), copy=True)
        vals[i + 1:] = vals[i + 1:] * (1 + rng.normal(0, 0.05, len(vals) - i - 1))
        tampered[col] = vals
    X2 = _features(tampered)
    same = np.allclose(np.nan_to_num(X[i]), np.nan_to_num(X2[i]))
    return same


def random_walk_control(vol, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, vol, n)
    c = 1.1 * np.exp(np.cumsum(ret))
    o = np.r_[c[0], c[:-1]]
    spread = np.abs(rng.normal(0, vol * 0.5, n))
    h = np.maximum(o, c) + spread
    low = np.minimum(o, c) - spread
    t = np.arange(n, dtype=float) * 3600
    X = engine.features(o, h, low, c, t)
    return engine.walk_forward_raw(X, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    args = ap.parse_args()

    fetch = get_fetcher(None)

    print("1) FEATURE CAUSALITY (features[i] must ignore candles > i)")
    ok = True
    for pair in args.pairs.split(","):
        try:
            df = fetch(pair, "1h", args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"   {pair}: fetch failed: {exc}")
            continue
        same = causality_check(df)
        ok &= same
        print(f"   {pair}: {'causal OK' if same else 'LEAK!'}")
    print(f"   -> {'PASS' if ok else 'FAIL'}\n")

    print("2) SYNTHETIC RANDOM WALK through the same engine (expect ~50%)")
    vols, reals = [], []
    for pair in args.pairs.split(","):
        try:
            df = fetch(pair, "1h", args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"   {pair}: fetch failed: {exc}")
            continue
        c = df["close"].to_numpy(float)
        r = np.diff(np.log(c))
        vols.append(float(np.std(r)))
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        low = df["low"].to_numpy(float)
        t = np.array([x.timestamp() for x in df.index], float)
        reals.append(engine.walk_forward_raw(engine.features(o, h, low, c, t), c))

    conf = np.concatenate([r["conf"] for r in reals])
    correct = np.concatenate([r["correct"] for r in reals])
    print(f"   REAL pooled floor 0.70: {100*correct[conf>=0.70].mean():.2f}%  "
          f"n={int((conf>=0.70).sum())}")

    rw_conf, rw_correct = [], []
    for k in range(6):
        raw = random_walk_control(vols[k % len(vols)], n=5000, seed=k)
        if raw:
            rw_conf.append(raw["conf"])
            rw_correct.append(raw["correct"])
    rw_conf = np.concatenate(rw_conf)
    rw_correct = np.concatenate(rw_correct)
    for floor in (0.60, 0.70):
        sel = rw_conf >= floor
        n = int(sel.sum())
        if n == 0:
            print(f"   RANDOM floor {floor}: no signals")
            continue
        wins = int(rw_correct[sel].sum())
        lo = readiness.wilson_lower_bound(wins, n) or 0.0
        print(f"   RANDOM floor {floor}: {100*wins/n:5.2f}%  n={n:5d}  "
              f"wilson_lo={100*lo:5.2f}%  "
              f"{'EDGE (BAD - harness leak!)' if lo > BREAK_EVEN else 'no edge (good)'}")


if __name__ == "__main__":
    main()
