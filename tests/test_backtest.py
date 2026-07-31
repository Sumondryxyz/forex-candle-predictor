import numpy as np
import pandas as pd
import pytest

from predictor.backtest import (
    accuracy_by_confidence,
    breakeven_accuracy,
    roi_pct,
    summary,
)


def _results(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    prob = rng.uniform(0.3, 0.9, n)
    actual = rng.integers(0, 2, n)
    df = pd.DataFrame({"prob_up": prob, "actual": actual})
    df["pred"] = (df["prob_up"] >= 0.5).astype(int)
    df["confidence"] = np.maximum(df["prob_up"], 1 - df["prob_up"])
    df["correct"] = (df["pred"] == df["actual"]).astype(int)
    return df


def test_breakeven_accuracy_matches_payout():
    assert breakeven_accuracy(0.80) == 1 / 1.8
    assert roi_pct(breakeven_accuracy(0.80), 0.80) == pytest.approx(0)


def test_roi_sign_follows_breakeven():
    assert roi_pct(0.60) > 0
    assert roi_pct(0.50) < 0


def test_confidence_table_is_monotone_in_coverage():
    table = accuracy_by_confidence(_results())
    assert table["signals"].is_monotonic_decreasing
    assert table.loc[0, "coverage_pct"] == 100.0


def test_summary_reports_sample_count():
    s = summary(_results(150))
    assert s["samples"] == 150
    assert 0 <= s["overall_accuracy_pct"] <= 100
