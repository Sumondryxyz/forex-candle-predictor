import numpy as np

from predictor import engine


def _candles(n: int = 1200, seed: int = 0):
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, n))
    high = close + rng.uniform(0, 2e-4, n)
    low = close - rng.uniform(0, 2e-4, n)
    open_ = np.r_[close[0], close[:-1]]
    times = 1_700_000_000 + np.arange(n) * 60
    rows = [
        {"open": open_[i], "high": high[i], "low": low[i], "close": close[i], "time": int(times[i])}
        for i in range(n)
    ]
    return rows


def test_arrays_unpack_ohlc_and_time():
    rows = _candles(10)
    o, h, low, c, t = engine.arrays(rows)
    assert o.shape == h.shape == low.shape == c.shape == t.shape == (10,)
    assert c[-1] == rows[-1]["close"]
    assert t[0] == rows[0]["time"]


def test_features_are_causal_and_finite():
    rows = _candles(400)
    o, h, low, c, t = engine.arrays(rows)
    X = engine.features(o, h, low, c, t)
    assert X.shape[0] == 400
    assert X.shape[1] >= 22
    assert np.isfinite(X).all()
    # Truncating the series must not change an already-closed candle's row.
    truncated = engine.features(o[:300], h[:300], low[:300], c[:300], t[:300])[-1]
    np.testing.assert_allclose(truncated, X[299])


def test_train_and_prob_are_calibrated_probabilities():
    rows = _candles(1500)
    o, h, low, c, t = engine.arrays(rows)
    X = engine.features(o, h, low, c, t)
    model = engine.train(X, c)
    assert model is not None
    p = model.prob(X[-1:])
    assert p.shape == (1,)
    assert 0.0 <= float(p[0]) <= 1.0


def test_train_needs_enough_history():
    rows = _candles(100)
    o, h, low, c, t = engine.arrays(rows)
    X = engine.features(o, h, low, c, t)
    assert engine.train(X, c) is None


def test_walk_forward_reports_honest_metrics():
    rows = _candles(1500, seed=2)
    o, h, low, c, t = engine.arrays(rows)
    X = engine.features(o, h, low, c, t)
    report = engine.walk_forward(X, c, min_conf=0.55, payout=0.8)
    assert report is not None
    assert report["signals"] >= 0
    assert 0 <= report["coverage"] <= 100


def test_numpy_fallback_backend_still_works(monkeypatch):
    # Termux has no scikit-learn; the numpy ensemble must remain a working path.
    monkeypatch.setattr(engine, "_LINEAR", False)
    rows = _candles(1500, seed=5)
    o, h, low, c, t = engine.arrays(rows)
    X = engine._features_numpy(o, h, low, c, t)
    model = engine.train(X, c)
    assert isinstance(model, engine.Ensemble)
    assert 0.0 <= float(model.prob(X[-1:])[0]) <= 1.0


def test_pandas_labels_stay_aligned_with_the_pandas_or_numpy_backend():
    # The pandas backend derives both features and labels from the same frame, so
    # every training row must have a real next-candle direction to learn from.
    rows = _candles(800, seed=7)
    o, h, low, c, t = engine.arrays(rows)
    X = engine.features(o, h, low, c, t)
    y, usable = engine.labels(c)
    assert X.shape[0] == len(c) == len(y) == len(usable)
    assert not usable[-1]  # last candle has no successor


def test_linear_model_is_calibrated_when_enough_history(monkeypatch):
    # A calibrated linear model must still expose prob() in [0, 1]; this guards
    # the scikit-learn >=1.6 calibration path (FrozenEstimator, not cv="prefit").
    monkeypatch.setattr(engine, "_CALIBRATE", True)
    monkeypatch.setattr(engine, "_CALIB_MIN", 300)
    rows = _candles(1500, seed=11)
    o, h, low, c, t = engine.arrays(rows)
    X = engine.features(o, h, low, c, t)
    model = engine.train(X, c)
    assert isinstance(model, engine.LinearModel)
    p = model.prob(X[-5:])
    assert p.shape == (5,)
    assert np.all((p >= 0) & (p <= 1))

