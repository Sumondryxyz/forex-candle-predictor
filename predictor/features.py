"""Technical feature engineering for next-candle direction prediction.

Every feature is computed from information available at the close of the
current candle only, so the dataset contains no look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "body_ratio", "upper_wick", "lower_wick", "range_z",
    "ema_fast_dist", "ema_slow_dist", "ema_cross",
    "rsi_14", "rsi_slope", "macd_hist", "macd_slope",
    "bb_pos", "atr_pct", "stoch_k", "stoch_d",
    "vol_ratio", "streak", "hour_sin", "hour_cos",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _streak(direction: pd.Series) -> pd.Series:
    """Signed length of the current run of consecutive up/down candles."""
    out = np.zeros(len(direction))
    run = 0
    prev = 0
    for i, d in enumerate(direction.to_numpy()):
        run = run + 1 if d == prev and d != 0 else 1
        prev = d
        out[i] = run * d
    return pd.Series(out, index=direction.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]
    rng = (high - low).replace(0, np.nan)
    atr = _atr(df)
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    lowest14 = low.rolling(14).min()
    highest14 = high.rolling(14).max()
    stoch_k = 100 * (close - lowest14) / (highest14 - lowest14).replace(0, np.nan)
    rsi = _rsi(close)
    hour = df.index.hour + df.index.minute / 60

    out = pd.DataFrame(index=df.index)
    out["ret_1"] = close.pct_change()
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    # Doji/flat candles have zero range; treat their shape features as neutral.
    out["body_ratio"] = ((close - open_) / rng).fillna(0)
    out["upper_wick"] = ((high - close.combine(open_, max)) / rng).fillna(0)
    out["lower_wick"] = ((close.combine(open_, min) - low) / rng).fillna(0)
    rng0 = (high - low)
    out["range_z"] = ((rng0 - rng0.rolling(50).mean()) / rng0.rolling(50).std().replace(0, np.nan)).fillna(0)
    out["ema_fast_dist"] = (close - ema_fast) / close
    out["ema_slow_dist"] = (close - ema_slow) / close
    out["ema_cross"] = (ema_fast - ema_slow) / close
    out["rsi_14"] = rsi
    out["rsi_slope"] = rsi.diff(3)
    out["macd_hist"] = macd_hist / close
    out["macd_slope"] = macd_hist.diff(3) / close
    bb_pos = (close - sma20) / (2 * std20).replace(0, np.nan)
    out["bb_pos"] = bb_pos.mask(std20 == 0, 0)
    out["atr_pct"] = atr / close
    out["stoch_k"] = stoch_k.fillna(50)
    out["stoch_d"] = stoch_k.fillna(50).rolling(3).mean()
    # Spot forex feeds usually report zero volume; neutralise the feature then.
    vol = df["volume"].replace(0, np.nan)
    out["vol_ratio"] = (vol / vol.rolling(50).mean()).fillna(1.0)
    out["streak"] = _streak(np.sign(close - open_).astype(int))
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features X and label y.

    The label mirrors a binary-option entry taken at the current close: 1 when
    the next candle closes higher than the current close. Unchanged closes are
    dropped because they are neither a win nor a loss.
    """
    X = build_features(df)
    change = df["close"].shift(-1) - df["close"]
    y = (change > 0).astype(int).where(change != 0)
    frame = X.join(y.rename("target")).dropna()
    return frame[FEATURE_COLUMNS], frame["target"].astype(int)
