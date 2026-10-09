#!/usr/bin/env python3
"""Next-candle engine, shared by the CLI and the web app.

Two backends behind one interface:

* **linear** (default when pandas + scikit-learn are importable): 35 causal
  features and a regularised logistic regression. Measured walk-forward, this
  beats gradient-boosted trees on 1-minute FX: trees overfit the noise and sit at
  ~51%, the linear model keeps a real edge once you filter on confidence.
* **numpy** (fallback, e.g. Termux without pandas): 22 causal features and a
  bagged logistic ensemble, calibrated on a held-out tail. Numpy-only so nothing
  has to compile on a phone.

Both expose the same surface: ``features``, ``train``, ``walk_forward`` and a
model with ``.prob(X)``. A signal is never a trade — the engine only predicts.
"""
from __future__ import annotations

import os

import numpy as np

try:  # pandas powers the richer feature set; absent on a bare Termux install.
    import pandas as pd

    _PANDAS = True
except ImportError:  # pragma: no cover - platform dependent
    pd = None
    _PANDAS = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _SKLEARN = True
except ImportError:  # pragma: no cover - platform dependent
    _SKLEARN = False

_LINEAR = _PANDAS and _SKLEARN
_LR_C = float(os.getenv("MODEL_C", "0.1"))
# Isotonic-calibrate the linear model on a held-out tail so a reported 0.60
# confidence means roughly a 60% hit rate (signals are filtered on confidence).
_CALIBRATE = os.getenv("MODEL_CALIBRATE", "1").lower() not in {"0", "false", "no"}
_CALIB_MIN = int(os.getenv("MODEL_CALIB_MIN", "1500"))



def ema(values, span):
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rolling(values, window, fn):
    out = np.zeros_like(values)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = fn(values[lo : i + 1])
    return out


def rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = ema(np.clip(delta, 0, None), period)
    loss = ema(np.clip(-delta, 0, None), period)
    rs = gain / np.where(loss == 0, 1e-12, loss)
    return 100 - 100 / (1 + rs)


def _features_numpy(o, h, low, c, times=None):
    """22 causal features, numpy only (the Termux path)."""
    eps = 1e-12
    rng = np.where(h - low == 0, eps, h - low)
    safe_c = np.where(c == 0, eps, c)

    ret1 = np.diff(c, prepend=c[0]) / safe_c
    ret3 = (c - np.roll(c, 3)) / safe_c
    ret5 = (c - np.roll(c, 5)) / safe_c
    ret10 = (c - np.roll(c, 10)) / safe_c

    ema_fast, ema_slow = ema(c, 5), ema(c, 20)
    ema_slowest = ema(c, 50)
    macd = ema(c, 12) - ema(c, 26)
    macd_hist = macd - ema(macd, 9)
    r = rsi(c)
    r_fast = rsi(c, 5)

    true_range = np.maximum(h - low, np.abs(np.diff(c, prepend=c[0])))
    atr = ema(true_range, 14)
    safe_atr = np.where(atr == 0, eps, atr)

    sma20 = rolling(c, 20, np.mean)
    std20 = rolling(c, 20, np.std)
    boll = (c - sma20) / np.where(std20 == 0, eps, std20)

    hi20 = rolling(h, 20, np.max)
    lo20 = rolling(low, 20, np.min)
    pos20 = (c - lo20) / np.where(hi20 - lo20 == 0, eps, hi20 - lo20)

    body = (c - o) / rng
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - low) / rng
    body_atr = (c - o) / safe_atr
    range_atr = rng / safe_atr

    up = (c > o).astype(float)
    streak = np.zeros_like(c)
    for i in range(1, len(c)):
        streak[i] = streak[i - 1] + 1 if up[i] == up[i - 1] else 0
    streak *= np.where(up == 1, 1, -1)

    vol_ratio = rolling(np.abs(ret1), 5, np.mean) / (rolling(np.abs(ret1), 30, np.mean) + eps)

    if times is None:
        sin_t = np.zeros_like(c)
        cos_t = np.zeros_like(c)
    else:
        frac = (np.asarray(times, float) % 86400) / 86400 * 2 * np.pi
        sin_t, cos_t = np.sin(frac), np.cos(frac)

    cols = [
        ret1 * 1e4, ret3 * 1e4, ret5 * 1e4, ret10 * 1e4,
        body, upper, lower, body_atr, range_atr,
        (c - ema_fast) / safe_atr, (c - ema_slow) / safe_atr, (c - ema_slowest) / safe_atr,
        (ema_fast - ema_slow) / safe_atr,
        (r - 50) / 50, (r_fast - 50) / 50, macd_hist / safe_atr,
        boll, pos20 - 0.5, streak / 5, vol_ratio - 1,
        sin_t, cos_t,
    ]
    X = np.column_stack(cols)
    X[:50] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _features_pandas(o, h, low, c, times, v=None):
    """35 causal features, reusing ``predictor.features`` (one source of truth)."""
    from .features import FEATURE_COLUMNS, build_features

    idx = (
        pd.to_datetime(np.asarray(times, float), unit="s", utc=True)
        if times is not None
        else pd.RangeIndex(len(c))
    )
    df = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "volume": np.zeros(len(c)) if v is None else np.asarray(v, float),
        },
        index=idx,
    )
    X = build_features(df)[FEATURE_COLUMNS].to_numpy(float)
    X[:50] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def backend_name() -> str:
    """Which model backend the engine will use in this environment."""
    return "linear" if _LINEAR else "numpy"


def features(o, h, low, c, times=None, v=None):
    """Causal features: row i only uses candles up to and including i."""
    if _PANDAS:
        return _features_pandas(o, h, low, c, times, v)
    return _features_numpy(o, h, low, c, times)


def fit_one(X, y, epochs=600, lr=0.15, l2=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / std
    w = rng.normal(0, 0.01, Z.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        err = p - y
        w -= lr * (Z.T @ err / len(y) + l2 * w)
        b -= lr * err.mean()
    return w, b, mean, std


def raw_prob(member, X):
    w, b, mean, std = member
    z = (X - mean) / std
    return 1 / (1 + np.exp(-np.clip(z @ w + b, -30, 30)))


class Ensemble:
    """Bagged logistic regressions plus a bin-based probability calibration."""

    def __init__(self, members, calib=None):
        self.members = members
        self.calib = calib or {}

    def raw(self, X):
        X = np.atleast_2d(X)
        return np.mean([raw_prob(m, X) for m in self.members], axis=0)

    def prob(self, X):
        p = self.raw(X)
        if not self.calib:
            return p
        edges, values = self.calib["edges"], self.calib["values"]
        idx = np.clip(np.digitize(p, edges) - 1, 0, len(values) - 1)
        return values[idx]


class LinearModel:
    """Thin wrapper so the linear backend matches the ensemble's ``.prob`` API."""

    def __init__(self, clf):
        self.clf = clf

    def prob(self, X):
        return self.clf.predict_proba(np.atleast_2d(X))[:, 1]


def labels(c):
    y = (np.roll(c, -1) > c).astype(float)
    usable = np.roll(c, -1) != c
    usable[-1] = False
    return y, usable


def _train_numpy(X, c, seeds=5):
    """Train on the first 80%, calibrate on the last 20% (never seen in fit)."""
    y, usable = labels(c)
    idx = np.flatnonzero(usable)
    if len(idx) < 300:
        return None

    split = int(len(idx) * 0.8)
    fit_idx, cal_idx = idx[:split], idx[split:]

    members = []
    n = len(fit_idx)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        sample = rng.choice(fit_idx, size=n, replace=True) if s else fit_idx
        members.append(fit_one(X[sample], y[sample], seed=s))
    model = Ensemble(members)

    p_cal = model.raw(X[cal_idx])
    y_cal = y[cal_idx]
    edges = np.linspace(0.3, 0.7, 9)
    values = []
    for i in range(len(edges)):
        lo = edges[i]
        hi = edges[i + 1] if i + 1 < len(edges) else 1.1
        in_bin = (p_cal >= lo) & (p_cal < hi)
        values.append(y_cal[in_bin].mean() if in_bin.sum() >= 15 else np.nan)
    values = np.array(values, float)
    centres = np.append(edges[1:], 1.0)
    values = np.where(np.isnan(values), np.clip(centres - 0.02, 0.05, 0.95), values)
    model.calib = {"edges": edges, "values": values}
    return model


def _train_linear(X, c):
    y, usable = labels(c)
    idx = np.flatnonzero(usable)
    if len(idx) < 300 or len(np.unique(y[idx])) < 2:
        return None
    base = make_pipeline(StandardScaler(), LogisticRegression(C=_LR_C, max_iter=2000))
    if _CALIBRATE and len(idx) >= _CALIB_MIN:
        split = int(len(idx) * 0.8)
        fit_idx, cal_idx = idx[:split], idx[split:]
        if len(np.unique(y[fit_idx])) >= 2 and len(np.unique(y[cal_idx])) >= 2:
            base.fit(X[fit_idx], y[fit_idx])
            try:
                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.frozen import FrozenEstimator

                cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
                cal.fit(X[cal_idx], y[cal_idx])
                return LinearModel(cal)
            except Exception:  # noqa: BLE001 - fall back to the raw scores
                base.fit(X[fit_idx], y[fit_idx])
                return LinearModel(base)
    base.fit(X[idx], y[idx])
    return LinearModel(base)


def train(X, c, seeds=5):
    """Fit the backend's model on the candles passed in."""
    if _LINEAR:
        return _train_linear(X, c)
    return _train_numpy(X, c, seeds=seeds)


def walk_forward_raw(X, c, train_size=None):
    """Per-observation out-of-sample confidences and correctness, in time order.

    Same fixed rolling-window scheme as :func:`walk_forward`, but returns the raw
    arrays so a caller can split them by time (e.g. choose a threshold on an
    early slice and measure it on a later one). Returns ``None`` if too short.
    """
    y, usable = labels(c)
    idx = np.flatnonzero(usable)
    n = len(idx)
    if n < 800:
        return None

    window = train_size or max(600, n // 2)
    step = max(50, window // 6)
    confs, corrects = [], []
    for start in range(window, n, step):
        train_idx = idx[start - window : start]
        test_idx = idx[start : start + step]
        model = train(X[train_idx], c[train_idx])
        if model is None or len(test_idx) == 0:
            continue
        p = model.prob(X[test_idx])
        confs.append(np.maximum(p, 1 - p))
        corrects.append((p > 0.5).astype(float) == y[test_idx])

    if not confs:
        return None
    return {
        "conf": np.concatenate(confs),
        "correct": np.concatenate(corrects).astype(float),
    }


SWEEP_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def summarise(raw, min_conf=0.60, payout=0.80, thresholds=SWEEP_THRESHOLDS):
    """Headline stats at ``min_conf`` plus the sweep, from raw walk-forward arrays.

    Kept separate from :func:`walk_forward` so a caller can hold onto the raw
    ``conf``/``correct`` arrays and re-threshold later without retraining.
    """
    conf = raw["conf"]
    correct = raw["correct"]

    def stats(threshold):
        picked = conf >= threshold
        n = int(picked.sum())
        if n == 0:
            return {"min_conf": threshold, "signals": 0, "coverage": 0.0, "rate": None, "roi": None}
        rate = float(correct[picked].mean())
        return {
            "min_conf": threshold,
            "signals": n,
            "coverage": round(100 * n / len(conf), 2),
            "rate": round(100 * rate, 2),
            "roi": round(100 * (rate * payout - (1 - rate)), 2),
        }

    headline = stats(min_conf)
    return {
        "signals": headline["signals"],
        "rate": (headline["rate"] or 0.0),
        "coverage": headline["coverage"],
        "roi": (headline["roi"] or 0.0),
        "sweep": [stats(t) for t in thresholds],
    }


def walk_forward(X, c, min_conf=0.60, payout=0.80, train_size=None):
    """Honest out-of-sample check: train on a rolling past window, score the next.

    A fixed rolling window (rather than an expanding one) is both more honest
    about how the live feed retrains and measurably stronger on FX, where very
    old candles stop being representative. Returns the headline rate at
    ``min_conf`` plus a ``sweep`` table showing the hit rate at several
    confidence floors, so a caller can see where the edge actually lives.
    """
    raw = walk_forward_raw(X, c, train_size=train_size)
    if raw is None:
        return None
    return summarise(raw, min_conf=min_conf, payout=payout)


def arrays(candles):
    """Turn a list of candle dicts into parallel numpy arrays."""
    o = np.array([x["open"] for x in candles], float)
    h = np.array([x["high"] for x in candles], float)
    low = np.array([x["low"] for x in candles], float)
    c = np.array([x["close"] for x in candles], float)
    t = np.array([x["time"] for x in candles], float)
    return o, h, low, c, t


def volumes(candles):
    """Per-candle volume (0.0 when the source does not report it)."""
    return np.array([float(x.get("volume") or 0.0) for x in candles], float)
