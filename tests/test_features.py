import numpy as np
import pandas as pd
import pytest

from predictor.features import FEATURE_COLUMNS, build_dataset, build_features


@pytest.fixture
def candles() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min", tz="UTC")
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, len(idx)))
    high = close + rng.uniform(0, 2e-4, len(idx))
    low = close - rng.uniform(0, 2e-4, len(idx))
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]], "high": high, "low": low, "close": close, "volume": 0},
        index=idx,
    )


def test_features_have_expected_columns(candles):
    X = build_features(candles)
    assert list(X.columns) == FEATURE_COLUMNS


def test_flat_candles_do_not_produce_nan(candles):
    flat = candles.copy()
    flat[["open", "high", "low", "close"]] = 1.1
    X = build_features(flat).iloc[60:]
    assert not X.isna().any(axis=None)


def test_dataset_labels_next_close_direction(candles):
    X, y = build_dataset(candles)
    assert len(X) == len(y)
    ts = X.index[10]
    nxt = candles.index[candles.index.get_loc(ts) + 1]
    expected = int(candles.loc[nxt, "close"] > candles.loc[ts, "close"])
    assert int(y.loc[ts]) == expected


def test_features_use_no_future_information(candles):
    truncated = build_features(candles.iloc[:300]).iloc[-1]
    full = build_features(candles).iloc[299]
    pd.testing.assert_series_equal(truncated, full)
