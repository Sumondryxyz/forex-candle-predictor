#!/usr/bin/env python3
"""Hunt for an honest 1m/5m edge on months of Dukascopy history.

The TradingView feed caps 1m at ~8.7 days, which is too short to tell signal
from noise. This reads the pickle from the parallel Dukascopy fetcher and tests
several honest variants on the SAME resampled prices:

  A. baseline walk-forward (what the live feed does)
  B. confidence floors
  C. volatility-regime filter (trade only calm or only wild candles)
  D. session / hour-of-day filter
  E. pooled model across pairs

Every variant reports hit rate, Wilson 95% lower bound and a one-sided binomial
p-value against the 55.56% break-even. An edge is only real when the LOWER BOUND
clears break-even.

    python research_1m5m_hunt.py --pickle var/duka_1m.pkl --intervals 1m,5m
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from predictor import engine, readiness

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


def _features(df):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    t = np.array([x.timestamp() for x in df.index], float)
    return engine.features(o, h, low, c, t), c


def _verdict(wins, n):
    if n == 0:
        return "no signals", 0.0, 1.0
    lo = readiness.wilson_lower_bound(wins, n) or 0.0
    p = readiness.p_value(wins, n, BREAK_EVEN)
    tag = "EDGE" if lo > BREAK_EVEN else "not yet"
    return tag, lo, p


def report(label, wins, n):
    tag, lo, p = _verdict(wins, n)
    print(f"  {label:<28} n={n:>6}  hit={100*wins/max(n,1):>6.2f}%  "
          f"wilson_lo={100*lo:>6.2f}%  p={p:>6.3f}  {tag}")


def _test_indices(c):
    """Row indices of ``raw['conf']`` in time order (mirrors walk_forward_raw)."""
    _, usable = engine.labels(c)
    idx = np.flatnonzero(usable)
    n = len(idx)
    window = max(600, n // 2)
    step = max(50, window // 6)
    parts = [idx[s : s + step] for s in range(window, n, step)]
    return np.concatenate(parts) if parts else np.array([], int)


def variant_a(raw, floors):
    print("  -- A/B: baseline + confidence floors --")
    for f in floors:
        picked = raw["conf"] >= f
        report(f"conf>={f:.2f}", int(raw["correct"][picked].sum()), int(picked.sum()))


def variant_c(raw, feats, c):
    """Volatility-regime filter: split by ATR percentile of the last candle."""
    print("  -- C: volatility-regime filter (atr_pct column) --")
    from predictor.features import FEATURE_COLUMNS

    if "atr_pct" not in FEATURE_COLUMNS:
        print("    (atr_pct not in feature set)")
        return
    tidx = _test_indices(c)
    if len(tidx) != len(raw["conf"]):
        print("    (index alignment failed)")
        return
    a = feats[:, FEATURE_COLUMNS.index("atr_pct")][tidx]
    med = pd.Series(a).rolling(200, min_periods=50).median().to_numpy()
    for name, mask in (("calm", a <= med), ("wild", a > med)):
        picked = mask & (raw["conf"] >= 0.70)
        report(f"{name} + conf>=0.70", int(raw["correct"][picked].sum()), int(picked.sum()))


def variant_d(raw, df, c):
    print("  -- D: session / hour filter --")
    tidx = _test_indices(c)
    if len(tidx) != len(raw["conf"]):
        print("    (index alignment failed)")
        return
    hours = np.array([x.hour for x in df.index])[tidx]
    for name, lo, hi in (("asia 0-7", 0, 7), ("london 7-12", 7, 12),
                         ("ny 12-20", 12, 20), ("late 20-24", 20, 24)):
        picked = (hours >= lo) & (hours < hi) & (raw["conf"] >= 0.70)
        report(f"{name} + conf>=0.70", int(raw["correct"][picked].sum()), int(picked.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pickle", default="var/duka_1m.pkl")
    ap.add_argument("--intervals", default="1m,5m")
    ap.add_argument("--floors", default="0.60,0.65,0.70,0.75,0.80")
    args = ap.parse_args()

    with open(args.pickle, "rb") as fh:
        store = pickle.load(fh)
    floors = [float(x) for x in args.floors.split(",")]

    print(f"backend={engine.backend_name()}  break-even={100*BREAK_EVEN:.2f}%")
    print(f"pairs: {', '.join(f'{p}({len(d)})' for p, d in store.items())}\n")

    for interval in args.intervals.split(","):
        print(f"== {interval} ==")
        for pair, raw_1m in store.items():
            df = _resample(raw_1m, interval)
            if len(df) < 500:
                print(f" {pair}: too short ({len(df)})")
                continue
            feats, c = _features(df)
            raw = engine.walk_forward_raw(feats, c)
            if raw is None:
                print(f" {pair}: walk-forward unavailable")
                continue
            print(f" {pair}: {len(df)} candles, base-rate up={100*np.mean(c[1:] > c[:-1]):.1f}%")
            variant_a(raw, floors)
            variant_c(raw, feats, c)
            variant_d(raw, df, c)
        print()


if __name__ == "__main__":
    main()
