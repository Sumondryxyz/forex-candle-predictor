import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from predictor import model
from predictor.features import build_dataset


def _trending_candles(n: int = 2500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, n))
    high = close + rng.uniform(0, 2e-4, n)
    low = close - rng.uniform(0, 2e-4, n)
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]], "high": high, "low": low, "close": close, "volume": 0},
        index=idx,
    )


def test_walk_forward_predicts_only_unseen_candles():
    df = _trending_candles()
    results = model.walk_forward(df, train_size=1200, step=300, calibrate=False)
    X, _ = build_dataset(df)
    assert results.index.min() >= X.index[1200]
    assert set(results.columns) >= {"prob_up", "actual", "pred", "confidence", "correct"}
    assert results["confidence"].between(0.5, 1.0).all()


def test_pooled_training_accepts_multiple_pairs():
    datasets = [build_dataset(_trending_candles(seed=s)) for s in (1, 2)]
    clf = model.train_pooled(datasets, calibrate=False)
    proba = clf.predict_proba(datasets[0][0].head(5))
    assert proba.shape == (5, 2)


def test_predict_latest_returns_direction_and_confidence():
    df = _trending_candles()
    X, y = build_dataset(df)
    bundle = {"model": model.train(X, y, calibrate=False), "features": list(X.columns)}
    pred = model.predict_latest(bundle, df)
    assert pred.direction in {"UP", "DOWN"}
    assert 0.5 <= pred.confidence <= 1.0
    assert pred.timestamp == df.index[-1]


def test_sklearn_fallback_when_lightgbm_is_missing(monkeypatch):
    monkeypatch.setattr(model, "lgb", None)
    assert isinstance(model.make_classifier(), HistGradientBoostingClassifier)

    X, y = build_dataset(_trending_candles(600))
    clf = model.train(X, y, calibrate=False)
    assert clf.predict_proba(X.head(3)).shape == (3, 2)
