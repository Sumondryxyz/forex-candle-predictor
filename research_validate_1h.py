#!/usr/bin/env python3
"""Rigorous validation of the ONE shippable signal: 1h, conf >= 0.70.

Runs the live engine's exact rolling walk-forward on real TradingView 1h data
for the six majors, then stresses the claim:

  * pooled hit rate at the 0.70 floor vs the 55.56% break-even (Wilson lo + p)
  * first-half vs second-half stability (a real edge must not live in one half)
  * by-month stability
  * a random-walk control: the same pipeline on shuffled returns must find
    nothing, else the "edge" is an artifact of the harness

    python research_validate_1h.py --days 180
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from predictor import engine, readiness
from predictor.sources import get_fetcher

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
BREAK_EVEN = 1 / (1 + 0.80)


def _raw(df):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    t = np.array([x.timestamp() for x in df.index], float)
    X = engine.features(o, h, low, c, t)
    raw = engine.walk_forward_raw(X, c)
    if raw is None:
        return None
    # keep timestamps aligned to the scored observations
    _, usable = engine.labels(c)
    idx = np.flatnonzero(usable)
    window = max(600, len(idx) // 2)
    step = max(50, window // 6)
    ts = []
    for start in range(window, len(idx), step):
        ts.extend(idx[start:start + step])
    ts = np.array([df.index[i].timestamp() for i in ts[: len(raw["conf"])]], float)
    raw["ts"] = ts
    return raw


def _line(tag, conf, correct, floor=0.70):
    sel = conf >= floor
    n = int(sel.sum())
    if n == 0:
        return f"  {tag}: no signals"
    wins = int(correct[sel].sum())
    lo = readiness.wilson_lower_bound(wins, n) or 0.0
    p = readiness.p_value(wins, n, BREAK_EVEN)
    verdict = "EDGE" if lo > BREAK_EVEN else "not yet"
    return (f"  {tag:22s}: {100*wins/n:5.2f}%  n={n:5d}  "
            f"wilson_lo={100*lo:5.2f}%  p={p:.3f}  {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    args = ap.parse_args()

    fetch = get_fetcher(None)
    all_conf, all_correct, all_ts = [], [], []
    print(f"1h validation  backend={engine.backend_name()}  days={args.days}  "
          f"break-even {100*BREAK_EVEN:.2f}%\n")
    for pair in args.pairs.split(","):
        try:
            df = fetch(pair, "1h", args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"  {pair}: fetch failed: {exc}")
            continue
        raw = _raw(df)
        if raw is None:
            print(f"  {pair}: history too short ({len(df)} bars)")
            continue
        all_conf.append(raw["conf"])
        all_correct.append(raw["correct"])
        all_ts.append(raw["ts"])
        print(f"  {pair}: n={len(df)} bars  {_line(pair, raw['conf'], raw['correct']).strip()}")

    conf = np.concatenate(all_conf)
    correct = np.concatenate(all_correct)
    ts = np.concatenate(all_ts)
    order = np.argsort(ts)
    conf, correct, ts = conf[order], correct[order], ts[order]

    print(f"\nPOOLED (all pairs):\n{_line('all', conf, correct)}")
    for floor in (0.60, 0.65, 0.70, 0.75):
        print(_line(f"floor {floor}", conf, correct, floor))

    mid = len(conf) // 2
    print("\nSTABILITY:")
    print(_line("first half", conf[:mid], correct[:mid]))
    print(_line("second half", conf[mid:], correct[mid:]))

    months = pd.to_datetime(ts, unit="s").strftime("%Y-%m")
    print("\nBY MONTH (floor 0.70):")
    for m in sorted(set(months)):
        sel = (months == m) & (conf >= 0.70)
        if sel.sum() >= 5:
            print(f"  {m}: {100*correct[sel].mean():5.2f}%  n={int(sel.sum())}")

    # random-walk control: shuffle the pooled confidences vs correctness pairing
    rng = np.random.default_rng(0)
    sel = conf >= 0.70
    hits = [correct[sel][rng.permutation(int(sel.sum()))].mean() for _ in range(200)]
    print(f"\nRANDOM-WALK CONTROL (shuffle labels, 200x): "
          f"mean={100*np.mean(hits):.2f}%  sd={100*np.std(hits):.2f}%  "
          f"(a real edge must sit far above this)")


if __name__ == "__main__":
    main()
