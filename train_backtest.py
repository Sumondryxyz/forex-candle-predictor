#!/usr/bin/env python3
"""Train a next-candle model and produce an honest walk-forward backtest report."""
from __future__ import annotations

import argparse
from pathlib import Path

from predictor import backtest, model
from predictor.features import build_dataset
from predictor.sources import get_fetcher

REPORTS = Path(__file__).resolve().parent / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="EURUSD")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--train-size", type=int, default=3000)
    ap.add_argument("--step", type=int, default=250)
    ap.add_argument(
        "--context-pairs",
        default="",
        help="comma separated pairs pooled into training, e.g. GBPUSD,AUDUSD",
    )
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--source", default=None, help="candle source: tradingview (default) or quotex")
    args = ap.parse_args()

    fetch_ohlc = get_fetcher(args.source)

    df = fetch_ohlc(args.pair, args.interval, args.days)
    print(f"{args.pair} {args.interval}: {len(df)} candles ({df.index[0]} -> {df.index[-1]})")

    context = []
    for other in filter(None, args.context_pairs.split(",")):
        other_df = fetch_ohlc(other.strip(), args.interval, args.days)
        print(f"  context {other.strip()}: {len(other_df)} candles")
        context.append(other_df)

    results = model.walk_forward(
        df,
        train_size=args.train_size,
        step=args.step,
        calibrate=not args.no_calibration,
        context=context,
    )
    report = backtest.format_report(args.pair, args.interval, results)
    print(report)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{args.pair.upper()}_{args.interval}.md").write_text(report)

    datasets = [build_dataset(df)] + [build_dataset(c) for c in context]
    final = model.train_pooled(datasets, calibrate=not args.no_calibration)
    metrics = backtest.summary(results)
    metrics["by_confidence"] = backtest.accuracy_by_confidence(results).to_dict("records")
    path = model.save(final, args.pair, args.interval, metrics)
    print(f"saved model -> {path}")


if __name__ == "__main__":
    main()
