"""Cross-pair confirmation: a second model must agree before a signal counts.

A per-pair model can be confidently wrong on its own history. Requiring the
signal to also agree with a model trained on every live pair of the same
interval filters exactly those cases: the pooled model has seen more of the
market, so a disagreement is a warning that the pair-local call is noise.

Measured on 15m (6 majors, ~53 days, walk-forward): confirmation lifted the
out-of-sample hit rate from 55.2% to 57.8% (n=277) and improved *every* pair.
It is opt-in (``CONSENSUS=1``) because it needs several pairs live at once and
trades signal count for accuracy: 348 signals became 277.

Nothing here places a trade; it only decides whether a signal is confirmed.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from . import engine

MIN_PAIRS = 2
# Rebuilding the pooled model is the expensive part, so reuse it for a while.
RETRAIN_SECONDS = float(os.getenv("CONSENSUS_RETRAIN", "300"))


def enabled() -> bool:
    return os.getenv("CONSENSUS", "0").strip().lower() in {"1", "true", "yes", "on"}


class _Registry:
    """Per-interval pool of live features, with a cached pooled model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._published: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self._models: dict[str, tuple[object, float]] = {}

    def publish(self, pair: str, interval: str, X: np.ndarray, c: np.ndarray) -> None:
        with self._lock:
            self._published.setdefault(interval, {})[pair] = (X, c)

    def _pooled_model(self, interval: str):
        cached = self._models.get(interval)
        if cached and time.time() - cached[1] < RETRAIN_SECONDS:
            return cached[0]
        series = list((self._published.get(interval) or {}).values())
        if len(series) < MIN_PAIRS:
            return None
        # Align on the shortest tail so every pair contributes the same span.
        length = min(len(c) for _, c in series)
        X = np.vstack([x[-length:] for x, _ in series])
        c = np.concatenate([cc[-length:] for _, cc in series])
        model = engine.train(X, c)
        self._models[interval] = (model, time.time())
        return model

    def agrees(
        self, pair: str, interval: str, x_row: np.ndarray, direction: str
    ) -> bool | None:
        """True/False if a pooled model can vote, else None (not enough pairs).

        ``engine.labels`` marks the final row unusable, so the pooled model is
        never trained on the candle it is asked to predict.
        """
        with self._lock:
            model = self._pooled_model(interval)
        if model is None:
            return None
        prob = float(model.prob(np.atleast_2d(x_row))[0])
        pooled_direction = "UP" if prob > 0.5 else "DOWN"
        return pooled_direction == direction


REGISTRY = _Registry()
