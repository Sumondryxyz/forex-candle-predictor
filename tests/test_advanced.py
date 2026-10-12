import numpy as np

from predictor import advanced, engine


def _candles(n=2500, seed=11):
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, n))
    high = close + rng.uniform(0, 2e-4, n)
    low = close - rng.uniform(0, 2e-4, n)
    open_ = np.r_[close[0], close[:-1]]
    times = 1_700_000_000 + np.arange(n) * 300
    return [
        {"open": open_[i], "high": high[i], "low": low[i], "close": close[i], "time": int(times[i])}
        for i in range(n)
    ]


def _matrix(n=2500, seed=11):
    rows = _candles(n, seed)
    o, h, low, c, t = engine.arrays(rows)
    return engine.features(o, h, low, c, t), c


def test_available_reflects_the_local_stack():
    assert advanced.available() is True  # scikit-learn + pandas are required deps


def test_ensemble_returns_a_probability_in_range():
    X, c = _matrix()
    model = advanced.train(X, c)
    assert model is not None
    p = model.prob(X[-5:])
    assert p.shape == (5,)
    assert np.all((p >= 0) & (p <= 1))


def test_ensemble_falls_back_to_none_when_history_is_short():
    X, c = _matrix(n=500, seed=3)
    assert advanced.train(X, c) is None


def test_blend_weight_stays_within_bounds():
    X, c = _matrix()
    model = advanced.train(X, c)
    assert 0.0 <= model.w <= 1.0
