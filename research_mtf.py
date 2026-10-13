#!/usr/bin/env python3
"""Can higher-timeframe context rescue the 1m/5m signal?

The 1m model only ever sees ~5100 candles (TradingView's cap), so it learns on
a few days of one regime. The classic fix is to feed it the *trend* from slower
timeframes: a 1m candle that agrees with the 15m trend may be less of a coin
flip than one that fights it.

This measures baseline vs multi-timeframe (MTF) features with the same rolling
walk-forward, and reports a Wilson lower bound + binomial p-value so a result
is only called an edge once the bound clears break-even.

    python research_mtf.py --pairs EURUSD,GBPUSD --fast 1m --days 120
"""
from __future__ import annotations

import argparse

import numpy as np

from predictor import engine, readiness
from predictor.sources import get_fetcher

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
SLOWER = {"1m": "15m", "5m": "30m", "15m": "1h"}


def _ohlc(df):
    return (df["open"].to_numpy(float), df["high"].to_numpy(float),
            df["low"].to_numpy(float), df["close"].to_numpy(float),
            np.array([x.timestamp() for x in df.index], float))


def _htf_context(slow_df, fast_times, slow_period):
    """Slower-timeframe features aligned to each fast timestamp.

    Only *fully closed* slower candles are used: a 15m candle's close is a
    future price relative to a 1m candle that is still forming, so aligning to
    the candle merely *containing* the fast time would leak the answer and
    inflate the score (it did: 68% before this guard, vs the honest number
    below). A slower candle is eligible only once it has closed at or before
    the fast candle's open.
    """
    o, h, low, c, t = _ohlc(slow_df)
    Xs = engine.features(o, h, low, c, t)
    pos = np.searchsorted(t + slow_period, fast_times, side="right") - 1
    pos = np.clip(pos, 0, len(t) - 1)
    # Keep the causal columns that describe trend/position, drop clock features.
    return Xs[pos][:, :20]


def run(pairs, fast, days, min_conf, payout):
    fetch = get_fetcher(None)
    slow = SLOWER[fast]
    break_even = 1 / (1 + payout)
    print(f"backend={engine.backend_name()}  {fast} + {slow} context  days={days}  "
          f"floor={min_conf}  break-even={100 * break_even:.2f}%\n")
    agg = {"base": [0, 0], "mtf": [0, 0]}
    for pair in pairs:
        try:
            fdf = fetch(pair, fast, days)
            sdf = fetch(pair, slow, days)
        except Exception as exc:  # noqa: BLE001
            print(f"{pair}: fetch failed {exc}")
            continue
        o, h, low, c, t = _ohlc(fdf)
        Xb = engine.features(o, h, low, c, t)
        Xm = np.column_stack([Xb, _htf_context(sdf, t, {"1m": 900, "5m": 1800, "15m": 3600}[fast])])
        for label, X in (("base", Xb), ("mtf", Xm)):
            r = engine.walk_forward_raw(X, c)
            if r is None:
                continue
            m = r["conf"] >= min_conf
            n = int(m.sum())
            w = int(r["correct"][m].sum())
            agg[label][0] += w
            agg[label][1] += n
            if label == "base":
                line = f"{pair:>7} n={n:>4} base={100 * w / n if n else 0:5.1f}%"
            else:
                base_w, base_n = agg["base"]
                b = 100 * base_w / base_n if base_n else 0
                line += f"  mtf={100 * w / n if n else 0:5.1f}%  (delta {100 * w / n - b:+.1f})"
                print(line)
    print()
    for label in ("base", "mtf"):
        w, n = agg[label]
        if not n:
            continue
        lo = readiness.wilson_lower_bound(w, n)
        p = readiness.p_value(w, n, break_even)
        verdict = "EDGE" if (lo or 0) > break_even else "no edge"
        print(f"{label:>5}: n={n:>5} hit={100 * w / n:5.2f}%  "
              f"wilson_lo={100 * (lo or 0):5.2f}%  p={p:.3f}  -> {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--fast", default="1m")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--payout", type=float, default=0.80)
    args = ap.parse_args()
    run(args.pairs.split(","), args.fast, args.days, args.min_conf, args.payout)
