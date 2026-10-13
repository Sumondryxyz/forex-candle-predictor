#!/usr/bin/env python3
"""Accuracy report: how often the engine's signals are right, per timeframe.

This is the honest scoreboard. For each pair and timeframe it runs the same
rolling walk-forward the live feed uses, then reports the hit rate at the
confidence floor together with a Wilson 95% lower bound and a one-sided
binomial p-value against the broker break-even. A rate above break-even is only
called an edge once the lower bound clears it.

    python research_accuracy.py                       # 6 majors, all timeframes
    python research_accuracy.py --intervals 1h --days 120
"""
from __future__ import annotations

import argparse

import numpy as np

from predictor import engine, readiness
from predictor.sources import get_fetcher

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]


def _measure(df, min_conf: float):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    t = np.array([x.timestamp() for x in df.index], float)
    X = engine.features(o, h, low, c, t)
    return engine.walk_forward_raw(X, c)


def run(pairs, intervals, days, min_conf, payout):
    fetch = get_fetcher(None)
    break_even = 1 / (1 + payout)
    print(f"backend={engine.backend_name()}  days={days}  min_conf={min_conf}  "
          f"payout={payout} -> break-even {100 * break_even:.2f}%\n")
    for interval in intervals:
        print(f"== {interval} ==")
        header = f"{'pair':>7} {'n':>6} {'signals':>8} {'hit%':>6} {'wilson_lo':>10} {'p':>7}  verdict"
        print(header)
        print("-" * len(header))
        agg_c = agg_n = 0
        for pair in pairs:
            try:
                df = fetch(pair, interval, days)
            except Exception as exc:  # noqa: BLE001
                print(f"{pair:>7}  fetch failed: {exc}")
                continue
            raw = _measure(df, min_conf)
            if raw is None:
                print(f"{pair:>7} {len(df):>6}  history too short")
                continue
            picked = raw["conf"] >= min_conf
            n = int(picked.sum())
            if n == 0:
                print(f"{pair:>7} {len(df):>6}        0  no signals at {min_conf}")
                continue
            wins = int(raw["correct"][picked].sum())
            lo = readiness.wilson_lower_bound(wins, n)
            p = readiness.p_value(wins, n, break_even)
            agg_c += wins
            agg_n += n
            verdict = "edge" if (lo or 0) > break_even else "not yet"
            print(f"{pair:>7} {len(df):>6} {n:>8} {100 * wins / n:>6.2f} "
                  f"{100 * (lo or 0):>10.2f} {p:>7.3f}  {verdict}")
        if agg_n:
            lo = readiness.wilson_lower_bound(agg_c, agg_n)
            p = readiness.p_value(agg_c, agg_n, break_even)
            verdict = "EDGE" if (lo or 0) > break_even else "not yet"
            print(f"{'ALL':>7} {'':>6} {agg_n:>8} {100 * agg_c / agg_n:>6.2f} "
                  f"{100 * (lo or 0):>10.2f} {p:>7.3f}  {verdict}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--intervals", default="5m,15m,30m,1h")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--payout", type=float, default=0.80)
    args = ap.parse_args()
    run(args.pairs.split(","), args.intervals.split(","), args.days,
        args.min_conf, args.payout)
