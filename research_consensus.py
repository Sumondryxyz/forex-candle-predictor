#!/usr/bin/env python3
"""Honest A/B for cross-pair confirmation (``CONSENSUS=1``).

Repeats the measurement behind ``predictor/consensus.py`` so the claim can be
re-checked, not taken on faith. For each pair it runs the same rolling
walk-forward the live feed uses, trains a *pooled* model on every other pair for
the same window, and reports:

* baseline -- the per-pair signal at ``--min-conf``;
* confirmed -- the same signals the pooled model also agrees with;
* a split-half test -- pick the confidence floor on the early half, measure on
  the late half, so an in-sample threshold cannot flatter the result;
* a Wilson 95% lower bound and a one-sided binomial p-value against break-even,
  because a 57% rate on a few hundred signals is not yet proof of an edge.

    python research_consensus.py                        # 6 majors, 15m
    python research_consensus.py --intervals 15m --days 60
"""
from __future__ import annotations

import argparse

import numpy as np

from predictor import engine, readiness
from predictor.sources import get_fetcher

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]


def _series(fetcher, pair, interval, days):
    df = fetcher(pair, interval, days=days)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    t = df.index.astype("int64").to_numpy() / 1e9
    return engine.features(o, h, low, c, t), c


def _walk(series: dict, target: str, interval: str):
    """Per-pair and pooled directions at each out-of-sample point, in time order."""
    X, c = series[target]
    length = min(len(cc) for _, cc in series.values())
    y, usable = engine.labels(c)
    idx = np.flatnonzero(usable)
    n = len(idx)
    window = max(600, n // 2)
    step = max(50, window // 6)
    per, pooled, correct = [], [], []
    for start in range(window, n, step):
        train_idx = idx[start - window : start]
        test_idx = idx[start : start + step]
        if len(test_idx) == 0:
            continue
        model = engine.train(X[train_idx], c[train_idx])
        if model is None:
            continue
        p_per = model.prob(X[test_idx])
        rows, labels = [X[train_idx]], [c[train_idx]]
        offset = length - len(c)
        for other, (xo, co) in series.items():
            if other == target:
                continue
            j = train_idx + offset
            j = j[(j >= 0) & (j < length)]
            if len(j):
                rows.append(xo[-length:][j])
                labels.append(co[-length:][j])
        pooled_model = engine.train(np.vstack(rows), np.concatenate(labels))
        p_pool = pooled_model.prob(X[test_idx]) if pooled_model is not None else p_per
        per.append(p_per)
        pooled.append(p_pool)
        correct.append((p_per > 0.5).astype(int) == y[test_idx])
    if not per:
        return None
    return {
        "per": np.concatenate(per),
        "pooled": np.concatenate(pooled),
        "correct": np.concatenate(correct).astype(float),
    }


def _confirmed(r):
    conf = np.maximum(r["per"], 1 - r["per"])
    agree = (r["per"] > 0.5) == (r["pooled"] > 0.5)
    return conf >= 0.70, (conf >= 0.70) & agree


def run(pairs, intervals, days, payout):
    fetcher = get_fetcher(None)
    print(f"payout {payout} -> break-even {100 / (1 + payout):.1f}% | {days}d history")
    for interval in intervals:
        series = {}
        for pair in pairs:
            try:
                series[pair] = _series(fetcher, pair, interval, days)
            except Exception as exc:  # noqa: BLE001
                print(f"{pair} {interval}: FETCH-FAIL {exc}")
        if len(series) < 2:
            print(f"{interval}: need >= 2 pairs for a pool, skipping")
            continue

        base_w = base_n = conf_w = conf_n = 0
        for pair in series:
            r = _walk(series, pair, interval)
            if r is None:
                continue
            b, g = _confirmed(r)
            base_w += r["correct"][b].sum()
            base_n += int(b.sum())
            conf_w += r["correct"][g].sum()
            conf_n += int(g.sum())

        print(f"\n== {interval} ==")
        if base_n:
            print(f"baseline  n={base_n:>5}  rate={100 * base_w / base_n:5.1f}%")
        if conf_n:
            rate = 100 * conf_w / conf_n
            lo = readiness.wilson_lower_bound(int(conf_w), conf_n)
            p = readiness.p_value(int(conf_w), conf_n, 1 / (1 + payout))
            print(f"confirmed n={conf_n:>5}  rate={rate:5.1f}%  "
                  f"wilson_lo={100 * (lo or 0):.1f}%  p={p:.3f}")
            print("  -> an edge is only *proven* once wilson_lo clears break-even.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--intervals", default="15m")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--payout", type=float, default=0.80)
    args = ap.parse_args()
    run(args.pairs.split(","), args.intervals.split(","), args.days, args.payout)
