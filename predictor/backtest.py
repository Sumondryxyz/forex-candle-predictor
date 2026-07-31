"""Reporting helpers for walk-forward results."""
from __future__ import annotations

import pandas as pd

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def accuracy_by_confidence(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(results)
    for t in THRESHOLDS:
        subset = results[results["confidence"] >= t]
        rows.append(
            {
                "min_confidence": t,
                "signals": len(subset),
                "coverage_pct": round(100 * len(subset) / total, 2) if total else 0.0,
                "accuracy_pct": round(100 * subset["correct"].mean(), 2) if len(subset) else None,
            }
        )
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> dict:
    baseline = max(results["actual"].mean(), 1 - results["actual"].mean())
    return {
        "samples": len(results),
        "overall_accuracy_pct": round(100 * results["correct"].mean(), 2),
        "always_majority_baseline_pct": round(100 * baseline, 2),
        "up_rate_pct": round(100 * results["actual"].mean(), 2),
    }


def format_report(pair: str, interval: str, results: pd.DataFrame) -> str:
    s = summary(results)
    table = accuracy_by_confidence(results).to_string(index=False)
    return (
        f"# Walk-forward backtest: {pair} {interval}\n\n"
        f"Out-of-sample samples: {s['samples']}\n"
        f"Overall accuracy: {s['overall_accuracy_pct']}%\n"
        f"Majority-class baseline: {s['always_majority_baseline_pct']}%\n"
        f"Up-candle rate: {s['up_rate_pct']}%\n\n"
        f"## Accuracy vs confidence filter\n```\n{table}\n```\n"
    )
