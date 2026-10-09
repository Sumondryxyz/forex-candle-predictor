"""Advanced ensemble signal: a calibrated linear model blended with a calibrated
gradient-boosted tree.

This is an optional upgrade over the single linear engine in ``engine.py``. Both
base models are isotonic-calibrated on a held-out tail, then the blend weight is
chosen on a further held-out slice, so the combination is measured rather than
guessed. Everything falls back to the linear engine when history is short or
scikit-learn is unavailable (Termux), so this never becomes a hard dependency.

Exposes the same surface the feed expects: ``train(X, c)`` returns a model with
``.prob(X)``, and ``available()`` says whether the richer path can run here.
"""
from __future__ import annotations

import numpy as np

from . import engine

try:  # the ensemble needs the pandas feature matrix and scikit-learn.
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _SKLEARN = True
except ImportError:  # pragma: no cover - platform dependent
    _SKLEARN = False

_MIN_ROWS = 1200


def available() -> bool:
    return bool(_SKLEARN and engine._PANDAS)


def _calibrate(base, Xf, yf, Xc, yc):
    """Fit ``base`` on the fit slice and wrap it in an isotonic calibrator."""
    base.fit(Xf, yf)
    if len(np.unique(yc)) < 2:
        return engine.LinearModel(base)
    try:
        cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
        cal.fit(Xc, yc)
        return engine.LinearModel(cal)
    except Exception:  # noqa: BLE001 - keep the raw scores if calibration fails
        return engine.LinearModel(base)


def _fit_linear(Xf, yf, Xc, yc):
    base = make_pipeline(StandardScaler(), LogisticRegression(C=engine._LR_C, max_iter=2000))
    return _calibrate(base, Xf, yf, Xc, yc)


def _fit_gbm(Xf, yf, Xc, yc):
    base = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=200,
        max_leaf_nodes=15,
        max_depth=4,
        min_samples_leaf=60,
        l2_regularization=5.0,
        early_stopping=False,
    )
    return _calibrate(base, Xf, yf, Xc, yc)


class Blend:
    """Weighted average of two calibrated models; ``w`` favours the linear one."""

    def __init__(self, linear, gbm, w):
        self.linear = linear
        self.gbm = gbm
        self.w = float(w)

    def prob(self, X):
        return self.w * self.linear.prob(X) + (1 - self.w) * self.gbm.prob(X)


def train(X, c):
    """Fit the ensemble on the candles given, or return ``None`` if too few.

    Split: first 60% fits both base models, the next 20% calibrates them, the
    last 20% picks the blend weight. None of those slices is reused for fitting,
    so the chosen weight is a genuine out-of-sample decision.
    """
    if not available():
        return None
    y, usable = engine.labels(c)
    idx = np.flatnonzero(usable)
    if len(idx) < _MIN_ROWS or len(np.unique(y[idx])) < 2:
        return None

    a = int(len(idx) * 0.6)
    b = int(len(idx) * 0.8)
    fit_idx, cal_idx, blend_idx = idx[:a], idx[a:b], idx[b:]
    if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[cal_idx])) < 2:
        return engine.train(X, c)

    lin = _fit_linear(X[fit_idx], y[fit_idx], X[cal_idx], y[cal_idx])
    gbm = _fit_gbm(X[fit_idx], y[fit_idx], X[cal_idx], y[cal_idx])

    p_lin = lin.prob(X[blend_idx])
    p_gbm = gbm.prob(X[blend_idx])
    yb = y[blend_idx]
    best_w, best_acc = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 11):
        acc = (((w * p_lin + (1 - w) * p_gbm) > 0.5).astype(float) == yb).mean()
        if acc > best_acc:
            best_acc, best_w = acc, float(w)
    return Blend(lin, gbm, best_w)
