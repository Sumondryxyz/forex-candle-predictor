#!/usr/bin/env python3
"""Accuracy report on long Dukascopy history (1m + resampled).

The point of this script is the one question TradingView could not answer: do
1m and 5m have an edge when the model gets *months* of history instead of the
~5.5 days its cap allows? It reads the pickle written by
``research_fetch_duka.py`` (raw 1m) and resamples it, so every timeframe is
measured on exactly the same prices, then prints the same honest scoreboard as
``research_accuracy.py`` (hit rate + Wilson lower bound + binomial p-value).

    python research_fetch_duka.py --pairs EURUSD,GBPUSD --days 120
    python research_duka_accuracy.py --pickle /tmp/duka_1m.pkl --intervals 1m,5m,15m,1h
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from predictor import engine, readiness

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
_RESAMPLE = {"1m": None, "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}


def _resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = _RESAMPLE[interval]
    if rule is None:
        return df
    return (df.resample(rule)
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"})
              .dropna())


def _measure(df: pd.DataFrame):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    t = np.array([x.timestamp() for x in df.index], float)
    X = engine.features(o, h, low, c, t)
    return engine.walk_forward_raw(X, c)


def run(store, intervals, min_conf, payout):
    break_even = 1 / (1 + payout)
    print(f"backend={engine.backend_name()}  min_conf={min_conf}  payout={payout} "
          f"-> break-even {100 * break_even:.2f}%")
    print(f"pairs: {', '.join(store)}\n")
    for interval in intervals:
        print(f"== {interval} ==")
        header = (f"{'pair':>7} {'candles':>8} {'signals':>8} {'hit%':>6} "
                  f"{'wilson_lo':>10} {'p':>7}  verdict")
        print(header)
        print("-" * len(header))
        agg_c = agg_n = 0
        for pair, raw_1m in store.items():
            df = _resample(raw_1m, interval)
            raw = _measure(df)
            if raw is None:
                print(f"{pair:>7} {len(df):>8}  history too short")
                continue
            picked = raw["conf"] >= min_conf
            n = int(picked.sum())
            if n == 0:
                print(f"{pair:>7} {len(df):>8} {0:>8}  no signals at {min_conf}")
                continue
            wins = int(raw["correct"][picked].sum())
            lo = readiness.wilson_lower_bound(wins, n)
            p = readiness.p_value(wins, n, break_even)
            agg_c += wins
            agg_n += n
            verdict = "edge" if (lo or 0) > break_even else "not yet"
            print(f"{pair:>7} {len(df):>8} {n:>8} {100 * wins / n:>6.2f} "
                  f"{100 * (lo or 0):>10.2f} {p:>7.3f}  {verdict}")
        if agg_n:
            lo = readiness.wilson_lower_bound(agg_c, agg_n)
            p = readiness.p_value(agg_c, agg_n, break_even)
            verdict = "EDGE" if (lo or 0) > break_even else "not yet"
            print(f"{'ALL':>7} {'':>8} {agg_n:>8} {100 * agg_c / agg_n:>6.2f} "
                  f"{100 * (lo or 0):>10.2f} {p:>7.3f}  {verdict}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pickle", default="/tmp/duka_1m.pkl")
    ap.add_argument("--intervals", default="1m,5m,15m,1h")
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--payout", type=float, default=0.80)
    args = ap.parse_args()
    with open(args.pickle, "rb") as fh:
        store = pickle.load(fh)
    run(store, args.intervals.split(","), args.min_conf, args.payout)
