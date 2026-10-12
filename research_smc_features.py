"""A/B: do Smart Money Concepts (SMC) features beat the baseline?

SMC = order blocks, fair-value gaps, liquidity sweeps, break of structure and
premium/discount zones. Same rolling walk-forward as the live app, same data,
only the feature matrix differs. Prints the hit rate at several floors for
baseline vs candidate so we ship it only if it wins out-of-sample.

Every SMC feature is causal: a swing is only "known" k bars after it forms, so
features at bar t never peek at future candles.
"""
from __future__ import annotations

import numpy as np

from predictor import engine
from predictor.sources.tradingview import fetch_ohlc

K = 3  # bars each side that confirm a swing high/low


def swing_marks(h, low, k=K):
    sh = np.zeros(len(h))
    sl = np.zeros(len(low))
    for i in range(k, len(h) - k):
        if h[i] == h[i - k : i + k + 1].max():
            sh[i] = 1
        if low[i] == low[i - k : i + k + 1].min():
            sl[i] = 1
    return sh, sl


def _last_known(marks, values, k):
    """Value of the most recent CONFIRMED swing at each bar (causal)."""
    out = np.full(len(values), np.nan)
    cur = np.nan
    for t in range(len(values)):
        idx = t - k
        if idx >= 0 and marks[idx]:
            cur = values[idx]
        out[t] = cur
    return out


def smc_features(o, h, low, c):
    """Causal SMC feature block (numpy, nan-filled then zeroed)."""
    n = len(c)
    sh, sl = swing_marks(h, low)
    last_sh = _last_known(sh, h, K)
    last_sl = _last_known(sl, low, K)

    eps = 1e-12
    atr = engine.ema(np.maximum(h - low, np.abs(np.diff(c, prepend=c[0]))), 14)

    # Distance to the controlling swing, in ATR units (structure proximity).
    dist_high = (last_sh - c) / np.where(atr == 0, eps, atr)
    dist_low = (c - last_sl) / np.where(atr == 0, eps, atr)

    # Premium/discount: 0 at the swing low, 1 at the swing high.
    rng = last_sh - last_sl
    prem_disc = (c - last_sl) / np.where(np.abs(rng) < eps, eps, rng)

    # Break of structure: close beyond the last confirmed swing.
    bos_up = (c > last_sh).astype(float)
    bos_dn = (c < last_sl).astype(float)

    # Liquidity sweep: wick pierced a swing then closed back inside, last 5 bars.
    swept_up = np.zeros(n)
    swept_dn = np.zeros(n)
    for t in range(5, n):
        if np.any(h[t - 5 : t] > last_sh[t]) and c[t] < last_sh[t]:
            swept_up[t] = 1.0
        if np.any(low[t - 5 : t] < last_sl[t]) and c[t] > last_sl[t]:
            swept_dn[t] = 1.0

    # Fair-value gap: 3-candle imbalance; distance to the most recent gap zone.
    fvg_pos = np.zeros(n)  # +1 bull gap, -1 bear gap active at t
    fvg_dist = np.zeros(n)
    last_gap = None  # (lo, hi, sign)
    for t in range(2, n):
        if low[t] > h[t - 2]:  # bullish imbalance
            last_gap = (h[t - 2], low[t], 1.0)
        elif h[t] < low[t - 2]:  # bearish imbalance
            last_gap = (h[t], low[t - 2], -1.0)
        if last_gap:
            lo, hi, sign = last_gap
            fvg_pos[t] = sign
            if lo <= c[t] <= hi:
                fvg_dist[t] = 0.0
            else:
                fvg_dist[t] = (min(abs(c[t] - lo), abs(c[t] - hi)) / np.where(atr[t] == 0, eps, atr[t])) * sign

    extra = np.column_stack([
        dist_high, dist_low, prem_disc, bos_up, bos_dn,
        swept_up, swept_dn, fvg_pos, fvg_dist,
    ])
    extra[: 60 + K] = 0.0
    return np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)


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
    floors = (0.55, 0.60, 0.65, 0.70, 0.75)
    print(f"data: {days} days | floors: " + "/".join(f"{f}" for f in floors))
    print(f"{'pair':<8}{'tf':<5}{'feat':<6}" + "".join(f"{f:>12}" for f in floors))
    wins = {f: 0 for f in floors}
    losses = {f: 0 for f in floors}
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
            Xc = np.column_stack([Xb, smc_features(o, h, low, c)])
            mb, mc = measure(Xb, c), measure(Xc, c)
            if mb is None or mc is None:
                continue
            rows += 1
            for label, m in (("base", mb), ("smc", mc)):
                cells = "".join(f"{m[f][1] if m[f][1] is not None else '-':>7}%{m[f][0]:>4}" for f in floors)
                print(f"{p:<8}{tf:<5}{label:<6}{cells}")
            for f in floors:
                a, b = mb[f][1], mc[f][1]
                if a is not None and b is not None and mb[f][0] >= 20 and mc[f][0] >= 20:
                    if b > a:
                        wins[f] += 1
                    elif b < a:
                        losses[f] += 1
    print(f"\nrows compared: {rows}")
    print(f"{'floor':<8}{'smc wins':>10}{'smc loses':>10}")
    for f in floors:
        print(f"{f:<8}{wins[f]:>10}{losses[f]:>10}")


if __name__ == "__main__":
    import sys

    pairs = (sys.argv[1] if len(sys.argv) > 1 else "EURUSD,GBPUSD,USDJPY,AUDUSD").split(",")
    intervals = (sys.argv[2] if len(sys.argv) > 2 else "5m,15m").split(",")
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    run(pairs, intervals, days)
