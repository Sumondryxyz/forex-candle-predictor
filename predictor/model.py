"""Gradient boosted model for next-candle direction."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

try:  # LightGBM has no wheels on Android/Termux, fall back to scikit-learn there.
    import lightgbm as lgb
except ImportError:  # pragma: no cover - platform dependent
    lgb = None

from .features import FEATURE_COLUMNS, build_dataset, build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "verbose": -1,
}

SKLEARN_PARAMS = {
    "learning_rate": 0.03,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "max_depth": 4,
    "min_samples_leaf": 60,
    "l2_regularization": 5.0,
    "early_stopping": False,
}


def make_classifier():
    """LightGBM when available, otherwise an equivalent scikit-learn model."""
    if lgb is None:
        return HistGradientBoostingClassifier(**SKLEARN_PARAMS)
    return lgb.LGBMClassifier(**PARAMS)


@dataclass
class Prediction:
    direction: str
    probability: float
    confidence: float
    timestamp: pd.Timestamp

    @property
    def is_tradeable(self) -> bool:
        return self.confidence >= 0.6


def train(X: pd.DataFrame, y: pd.Series, calibrate: bool = True):
    """Fit the classifier, optionally wrapping it in an isotonic calibrator.

    Calibration matters here because signals are filtered on confidence: an
    uncalibrated 0.70 score does not mean a 70% hit rate.
    """
    base = make_classifier()
    if not calibrate or len(X) < 800 or y.nunique() < 2:
        base.fit(X, y)
        return base
    cutoff = int(len(X) * 0.8)
    base.fit(X.iloc[:cutoff], y.iloc[:cutoff])
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calibrated.fit(X.iloc[cutoff:], y.iloc[cutoff:])
    return calibrated


def train_pooled(datasets: list[tuple[pd.DataFrame, pd.Series]], calibrate: bool = True):
    """Train one model on several pairs at once for a larger, more robust sample."""
    X = pd.concat([d[0] for d in datasets])
    y = pd.concat([d[1] for d in datasets])
    order = X.index.argsort()
    return train(X.iloc[order], y.iloc[order], calibrate=calibrate)


def model_path(pair: str, interval: str) -> Path:
    return MODEL_DIR / f"{pair.upper()}_{interval}.joblib"


def save(model, pair: str, interval: str, metrics: dict) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(pair, interval)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "metrics": metrics}, path)
    path.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    return path


def load(pair: str, interval: str):
    path = model_path(pair, interval)
    if not path.exists():
        raise FileNotFoundError(f"no trained model at {path}; run train_backtest.py first")
    return joblib.load(path)


def predict_latest(bundle: dict, df: pd.DataFrame) -> Prediction:
    """Predict the direction of the candle that follows the last closed candle."""
    features = build_features(df).iloc[[-1]][bundle["features"]]
    if features.isna().any(axis=None):
        raise ValueError("latest candle has incomplete indicator history")
    prob_up = float(bundle["model"].predict_proba(features)[0, 1])
    return Prediction(
        direction="UP" if prob_up >= 0.5 else "DOWN",
        probability=prob_up,
        confidence=max(prob_up, 1 - prob_up),
        timestamp=df.index[-1],
    )


def walk_forward(
    df: pd.DataFrame,
    train_size: int = 3000,
    step: int = 250,
    calibrate: bool = True,
    context: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Retrain periodically and predict only on data the model has never seen.

    ``context`` holds other pairs whose candles are pooled into each training
    window (restricted to the same time span) to enlarge the training sample.
    """
    X, y = build_dataset(df)
    extra = [build_dataset(other) for other in (context or [])]
    if len(X) <= train_size + step:
        train_size = max(int(len(X) * 0.6), 200)
        step = max(int(len(X) * 0.1), 50)
    rows = []
    for start in range(train_size, len(X) - 1, step):
        window = [(X.iloc[start - train_size : start], y.iloc[start - train_size : start])]
        lo, hi = window[0][0].index[0], window[0][0].index[-1]
        for xo, yo in extra:
            mask = (xo.index >= lo) & (xo.index <= hi)
            if mask.sum():
                window.append((xo[mask], yo[mask]))
        model = train_pooled(window, calibrate=calibrate)
        block = X.iloc[start : start + step]
        prob = model.predict_proba(block)[:, 1]
        rows.append(
            pd.DataFrame(
                {"prob_up": prob, "actual": y.iloc[start : start + step].to_numpy()},
                index=block.index,
            )
        )
    out = pd.concat(rows)
    out["pred"] = (out["prob_up"] >= 0.5).astype(int)
    out["confidence"] = np.maximum(out["prob_up"], 1 - out["prob_up"])
    out["correct"] = (out["pred"] == out["actual"]).astype(int)
    return out
