#!/usr/bin/env python3
"""Download months of Dukascopy 1m candles into a pickle for research.

    python research_fetch_duka.py --pairs EURUSD,GBPUSD --days 180 --out /tmp/duka.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time

from predictor.sources import dukascopy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--out", default="/tmp/duka_1m.pkl")
    ap.add_argument("--pause", type=float, default=0.6)
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    end = None
    if args.end:
        import datetime as dt
        end = dt.date.fromisoformat(args.end)

    store = {}
    for pair in args.pairs.split(","):
        t0 = time.time()
        try:
            df = dukascopy.fetch_ohlc(pair, "1m", args.days, end=end,
                                      pause=args.pause)
        except Exception as exc:  # noqa: BLE001
            print(f"{pair}: FAILED {exc}", flush=True)
            continue
        store[pair] = df
        # Save after each pair so a long, throttled download can be analysed
        # as soon as its first pair lands (and survives a mid-run kill).
        with open(args.out, "wb") as fh:
            pickle.dump(store, fh)
        print(f"{pair}: {len(df)} candles, {df.index[0]} .. {df.index[-1]} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {args.out} ({len(store)} pairs)", flush=True)


if __name__ == "__main__":
    main()
