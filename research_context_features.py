"""A/B: do multi-timeframe + volatility-regime features beat the baseline?

Same rolling walk-forward as the live app, same data, only the feature matrix
differs. Prints the hit rate at several confidence floors for baseline vs
candidate so we can decide honestly whether to ship.
"""
from __future__ import annotations

import numpy as np

from predictor import engine
from predictor.sources.tradingview import fetch_ohlc


def _base_plus(o, h, low, c, times):
    """Baseline 22 numpy features plus candidate context features (all causal)."""
    X = engine.features(o, h, low, c, times)
    eps = 1e-12

    # Volatility regime: current ATR vs its own longer-run average.
    tr = np.maximum(h - low, np.abs(np.diff(c, prepend=c[0])))
    atr = engine.ema(tr, 14)
    atr_long = engine.rolling(atr, 100, np.mean)
    vol_regime = (atr / np.where(atr_long == 0, eps, atr_long)) - 1

    # Band width (squeeze) and mean-reversion pressure.
    sma20 = engine.rolling(c, 20, np.mean)
    std20 = engine.rolling(c, 20, np.std)
    band = std20 / np.where(sma20 == 0, eps, sma20)

    # Kaufman efficiency ratio over 30 bars: trend vs chop.
    net = np.abs(c - np.roll(c, 30))
    path = engine.rolling(np.abs(np.diff(c, prepend=c[0])), 30, np.sum)
    eff = net / np.where(path == 0, eps, path)

    # Higher-timeframe momentum: returns over longer horizons.
    safe = np.where(c == 0, eps, c)
    ret30 = (c - np.roll(c, 30)) / safe
    ret60 = (c - np.roll(c, 60)) / safe

    # One-bar autocorrelation of returns (mean-reversion vs momentum).
    r1 = np.diff(c, prepend=c[0]) / safe
    ac1 = engine.rolling(r1 * np.roll(r1, 1), 30, np.mean) / (
        engine.rolling(r1 * r1, 30, np.mean) + eps
    )

    extra = np.column_stack([
        vol_regime, band, eff, ret30 * 1e4, ret60 * 1e4, ac1,
    ])
    extra[:60] = 0.0
    X = np.column_stack([X, np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)])
    return X


def measure(X, c, floors=(0.55, 0.60, 0.65, 0.70, 0.75)):
    raw = engine.walk_forward_raw(X, c)
    if raw is None:
        return None
    out = {}
    for f in floors:
        picked = raw["conf"] >= f
        out[f] = (int(picked.sum()), round(100 * float(raw["correct"][picked].mean()), 2)
                  if picked.sum() else None)
    return out


def run(pairs, intervals, days):
    print(f"data: {days} days | floors: 0.55/0.60/0.65/0.70/0.75")
    print(f"{'pair':<8}{'tf':<5}{'feat':<5}" + "".join(f"{f:>12}" for f in (0.55, 0.60, 0.65, 0.70, 0.75)))
    wins = {f: 0 for f in (0.55, 0.60, 0.65, 0.70, 0.75)}
    losses = {f: 0 for f in (0.55, 0.60, 0.65, 0.70, 0.75)}
    rows = 0
    for p in pairs:
        for tf in intervals:
            try:
                df = fetch_ohlc(p, tf, days=days)
            except Exception as exc:  # noqa: BLE001
                print(f"{p:<8}{tf:<5}FETCH-FAIL {exc}")
                continue
            o = df["open"].to_numpy(float)
            h = df["high"].to_numpy(float)
            low = df["low"].to_numpy(float)
            c = df["close"].to_numpy(float)
            t = df.index.astype("int64").to_numpy() / 1e9
            Xb = engine.features(o, h, low, c, t)
            Xc = _base_plus(o, h, low, c, t)
            mb = measure(Xb, c)
            mc = measure(Xc, c)
            if mb is None or mc is None:
                continue
            rows += 1
            for label, m, X in (("base", mb, Xb), ("new", mc, Xc)):
                cells = "".join(f"{m[f][1] if m[f][1] is not None else '-':>7}%{m[f][0]:>4}" for f in (0.55, 0.60, 0.65, 0.70, 0.75))
                print(f"{p:<8}{tf:<5}{label:<5}{cells}")
            for f in wins:
                a, b = mb[f][1], mc[f][1]
                if a is not None and b is not None and mb[f][0] >= 20 and mc[f][0] >= 20:
                    if b > a:
                        wins[f] += 1
                    elif b < a:
                        losses[f] += 1
    print(f"\nrows compared: {rows}")
    print(f"{'floor':<8}{'new wins':>10}{'new loses':>10}")
    for f, w in wins.items():
        print(f"{f:<8}{w:>10}{losses[f]:>10}")


if __name__ == "__main__":
    import sys

    pairs = (sys.argv[1] if len(sys.argv) > 1 else "EURUSD,GBPUSD,USDJPY,AUDUSD").split(",")
    intervals = (sys.argv[2] if len(sys.argv) > 2 else "5m,15m").split(",")
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    run(pairs, intervals, days)
