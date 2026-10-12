#!/usr/bin/env python3
"""Measure the engine's honest edge across pairs and intervals.

This is the research harness: it fetches candles from TradingView, runs the
same fixed rolling-window walk-forward the live model uses, and prints the
hit rate / ROI at several confidence floors. Use it to compare a candidate
change against the current engine on the SAME data before shipping it.

    python research_sweep.py                                  # default set
    python research_sweep.py --pairs EURUSD,GBPUSD --intervals 5m,15m
    python research_sweep.py --days 30 --json reports/sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from predictor import engine
from predictor.sources import get_fetcher


def _use_backend(name: str) -> None:
    """Optionally swap the model behind ``engine.train`` for the experiment."""
    if name == "default":
        return
    if name == "ensemble":
        from predictor import advanced

        if not advanced.available():
            raise SystemExit("ensemble backend unavailable in this environment")
        engine.train = advanced.train  # the harness only, never the live app
        return
    raise SystemExit(f"unknown backend {name!r} (use default or ensemble)")


def measure(df, min_conf: float, payout: float):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    t = np.array([x.timestamp() for x in df.index], float)
    X = engine.features(o, h, low, c, t)
    return engine.walk_forward(X, c, min_conf=min_conf, payout=payout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY")
    ap.add_argument("--intervals", default="5m,15m")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--payout", type=float, default=0.80)
    ap.add_argument("--source", default=None)
    ap.add_argument("--backend", default="default", help="default (linear) or ensemble")
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default="current")
    args = ap.parse_args()

    _use_backend(args.backend)
    fetch = get_fetcher(args.source)
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    shown = engine.backend_name() if args.backend == "default" else args.backend
    print(f"backend={shown}  tag={args.tag}  days={args.days}")
    header = f"{'pair':>7} {'int':>4} {'n':>6} {'signals':>8} {'cov%':>6} {'hit%':>6} {'ROI%':>7}"
    print(header)
    print("-" * len(header))

    payload = {"tag": args.tag, "backend": engine.backend_name(), "rows": []}
    for pair in pairs:
        for interval in intervals:
            try:
                df = fetch(pair, interval, args.days)
            except Exception as exc:  # noqa: BLE001 - keep sweeping other combos
                print(f"{pair:>7} {interval:>4}  fetch failed: {exc}")
                continue
            res = measure(df, args.min_conf, args.payout)
            if not res or not res["signals"]:
                print(f"{pair:>7} {interval:>4} {len(df):>6}  no signals at {args.min_conf}")
                continue
            print(
                f"{pair:>7} {interval:>4} {len(df):>6} {res['signals']:>8} "
                f"{res['coverage']:>6.1f} {res['rate']:>6.2f} {res['roi']:>7.2f}"
            )
            payload["rows"].append({"pair": pair, "interval": interval, "n": len(df), **res})

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
