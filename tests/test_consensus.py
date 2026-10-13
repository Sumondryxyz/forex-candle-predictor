import time

import numpy as np

from predictor import consensus


def _series(n: int = 1500, seed: int = 0, drift: float = 1e-5):
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(drift, 1e-4, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 2e-4, n)
    low = np.minimum(open_, close) - rng.uniform(0, 2e-4, n)
    times = np.arange(n, dtype=float) * 60 + (int(time.time()) - n * 60)
    from predictor import engine

    X = engine.features(open_, high, low, close, times, None)
    return X, close


def _reset():
    consensus.REGISTRY._published.clear()
    consensus.REGISTRY._models.clear()


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("CONSENSUS", raising=False)
    assert consensus.enabled() is False
    monkeypatch.setenv("CONSENSUS", "1")
    assert consensus.enabled() is True


def test_agrees_is_none_with_a_single_pair():
    _reset()
    X, c = _series()
    consensus.REGISTRY.publish("EURUSD", "5m", X, c)
    assert consensus.REGISTRY.agrees("EURUSD", "5m", X[-1], "UP") is None


def test_agrees_returns_a_boolean_once_two_pairs_are_live():
    _reset()
    for i, pair in enumerate(("EURUSD", "GBPUSD")):
        X, c = _series(seed=i)
        consensus.REGISTRY.publish(pair, "5m", X, c)
    verdict = consensus.REGISTRY.agrees("EURUSD", "5m", _series(seed=0)[0][-1], "UP")
    assert isinstance(verdict, bool)
