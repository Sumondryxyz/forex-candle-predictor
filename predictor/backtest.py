"""Reporting helpers for walk-forward results."""
from __future__ import annotations

import pandas as pd

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Typical binary-option payout on a winning trade; a loss costs the full stake.
DEFAULT_PAYOUT = 0.80


def roi_pct(accuracy: float, payout: float = DEFAULT_PAYOUT) -> float:
    """Expected return per unit staked at the given hit rate."""
    return 100 * (accuracy * payout - (1 - accuracy))


def breakeven_accuracy(payout: float = DEFAULT_PAYOUT) -> float:
    return 1 / (1 + payout)


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
                "roi_pct_per_trade": round(roi_pct(subset["correct"].mean()), 2) if len(subset) else None,
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
        "breakeven_accuracy_pct": round(100 * breakeven_accuracy(), 2),
    }


def format_report(pair: str, interval: str, results: pd.DataFrame) -> str:
    s = summary(results)
    table = accuracy_by_confidence(results).to_string(index=False)
    return (
        f"# Walk-forward backtest: {pair} {interval}\n\n"
        f"Out-of-sample samples: {s['samples']}\n"
        f"Overall accuracy: {s['overall_accuracy_pct']}%\n"
        f"Majority-class baseline: {s['always_majority_baseline_pct']}%\n"
        f"Up-candle rate: {s['up_rate_pct']}%\n"
        f"Break-even accuracy at {int(DEFAULT_PAYOUT * 100)}% payout: "
        f"{s['breakeven_accuracy_pct']}%\n\n"
        f"## Accuracy vs confidence filter\n```\n{table}\n```\n"
    )
